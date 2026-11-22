#!/usr/bin/env python3
"""Apply the MID360 fork's preintegration reset repair to its active catkin source."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

MARK = 'GO2_PREINTEGRATION_RESET_V1'

def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('源码结构与此补丁不符，定位失败：'+old[:85])
    return text.replace(old, new, 1)


def motion_feedback(files):
    """Honor the attitude weight in the correction fed back to preintegration."""
    result = dict(files)
    name = 'src/mapOptmization.cpp'
    s = result[name]
    mark = 'GO2_IMU_FEEDBACK_WEIGHT_V1'
    if mark in s:
        if 'double imuWeight = 0.1;' in s[s.index('    void publishOdometry()'):]:
            raise ValueError('姿态反馈补丁标记存在但仍有硬编码权重')
        return result
    a, b = s.index('    void publishOdometry()'), s.index('    void publishFrames()')
    block = s[a:b]
    block = replace_once(block, '            if (cloudInfo.imuAvailable == true)',
        '            // '+mark+'\n'
        '            // A synthesized six-axis attitude is not an independent correction.\n'
        '            // Keep initialization and gyro deskew; respect the configured fusion weight.\n'
        '            if (cloudInfo.imuAvailable && imuType == 1 && imuRPYWeight > 0.0)')
    block = replace_once(block, '                    double imuWeight = 0.1;',
        '                    double imuWeight = imuRPYWeight;')
    result[name] = s[:a]+block+s[b:]
    return result

def transform(files):
    result = dict(files)
    header = result['include/utility.h']
    # Only core Mat/Scalar, eigen, transpose and solve are used here.
    # The umbrella header also includes cvflann, whose macros clash with PCL FLANN.
    for include in ('#include <opencv/cv.h>', '#include <opencv2/opencv.hpp>'):
        if include in header:
            header = replace_once(header, include, '#include <opencv2/core.hpp>')
    result['include/utility.h'] = header
    reset_sources = [files[name] for name in PATHS if name != 'include/utility.h']
    if all(MARK in value for value in reset_sources):
        return motion_feedback(result)
    if any(MARK in value for value in reset_sources):
        raise ValueError('检测到不完整的旧补丁，请先恢复同一份备份')
    p = 'src/imuPreintegration.cpp'
    s = result[p]
    s = replace_once(s, '    void resetParams()\n    {',
        '    // '+MARK+'\n    unsigned long long imuResetId = 0;\n\n'
        '    void resetParams()\n    {\n        ++imuResetId;')
    s = replace_once(s, '            // initial bias\n',
        '            // Reset the state actually passed to predict(), including after failure.\n'
        '            prevState_ = gtsam::NavState(prevPose_, prevVel_);\n'
        '            prevStateOdom = prevState_;\n'
        '            // initial bias\n')
    s = replace_once(s, '            prevBias_ = gtsam::imuBias::ConstantBias();',
        '            prevBias_ = gtsam::imuBias::ConstantBias();\n'
        '            prevBiasOdom = prevBias_;')
    s = replace_once(s, '        pubImuOdometry.publish(odometry);',
        '        // The projection frontend already uses this slot as a reset counter.\n'
        '        odometry.pose.covariance[0] = static_cast<double>(imuResetId);\n'
        '        pubImuOdometry.publish(odometry);')
    result[p] = s
    p = 'msg/cloud_info.msg'
    result[p] = replace_once(result[p], 'int64 odomAvailable',
        'int64 odomAvailable\n\n# '+MARK+'\nuint64 imuResetId')
    p = 'src/imageProjection.cpp'
    s = result[p]
    s = replace_once(s, '        cloudInfo.odomAvailable = true;',
        '        // '+MARK+'\n'
        '        cloudInfo.imuResetId = static_cast<uint64_t>(startOdomMsg.pose.covariance[0]);\n'
        '        cloudInfo.odomAvailable = true;')
    result[p] = s
    p = 'src/mapOptmization.cpp'
    s = result[p]
    s = replace_once(s, '        static Eigen::Affine3f lastImuPreTransformation;',
        '        static Eigen::Affine3f lastImuPreTransformation;\n'
        '        // '+MARK+'\n'
        '        static uint64_t lastImuResetId = 0;\n'
        '        if (!cloudInfo.odomAvailable)\n'
        '            lastImuPreTransAvailable = false;')
    s = replace_once(s, '            if (lastImuPreTransAvailable == false)',
        '            if (!lastImuPreTransAvailable || lastImuResetId != cloudInfo.imuResetId)')
    s = replace_once(s, '                lastImuPreTransAvailable = true;',
        '                lastImuPreTransAvailable = true;\n'
        '                lastImuResetId = cloudInfo.imuResetId;')
    # On an uncorrected scan, the incremental output still needs this scan's
    # predicted back transform. Otherwise it reuses the preceding scan's back.
    s = replace_once(s, '    void scan2MapOptimization()\n    {',
        '    void scan2MapOptimization()\n    {\n'
        '        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);')
    result[p] = s
    return motion_feedback(result)

PATHS = ('src/imuPreintegration.cpp', 'msg/cloud_info.msg',
         'src/imageProjection.cpp', 'src/mapOptmization.cpp', 'include/utility.h')

def install(package, workspace):
    package, workspace = Path(package).resolve(), Path(workspace).resolve()
    if workspace / 'src' not in package.parents:
        raise ValueError('当前 lio_sam 不在指定 catkin 工作空间源码目录中')
    originals = {name: (package / name).read_text() for name in PATHS}
    updated = transform(originals)
    if originals == updated:
        print('重置修复已安装，继续编译确认。', flush=True)
    backup = Path(tempfile.mkdtemp(prefix='go2_preintegration_backup_', dir=str(Path.home())))
    for name in PATHS:
        dest = backup / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package / name, dest)
    (backup / 'manifest.json').write_text(json.dumps({
        'package': str(package),
        'sha256': {name: hashlib.sha256((package/name).read_bytes()).hexdigest() for name in PATHS}
    }, indent=2))
    print('原源码备份 '+str(backup), flush=True)
    try:
        for name in PATHS:
            target = package / name
            fd, temp = tempfile.mkstemp(dir=str(target.parent))
            try:
                with os.fdopen(fd, 'w') as stream:
                    stream.write(updated[name])
                shutil.copymode(target, temp)
                os.replace(temp, target)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
        # catkin_make resolves its workspace through PWD, even when cwd differs.
        build_env = dict(os.environ, PWD=str(workspace))
        subprocess.run(['catkin_make', '--directory', str(workspace),
                        '--pkg', 'lio_sam', '-j2', '-l2'],
                       cwd=str(workspace), env=build_env, check=True)
    except BaseException:
        for name in PATHS:
            shutil.copy2(backup / name, package / name)
        print('编译未完成，已恢复原源码。请保留编译错误，完成重新编译后再启动。', file=sys.stderr)
        raise
    print('预积分重置及增量里程计姿态反馈修复已编译完成。', flush=True)

if __name__ == '__main__':
    try:
        if len(sys.argv) != 3:
            raise ValueError('请通过 repair_mapping.sh 启动')
        install(*sys.argv[1:])
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        sys.exit(str(exc))
