# 多点路线保存与复用

导航运行时，在 Foxglove 中按顺序设置目标。等航点编号出现后，在另一终端保存；保存不改变运动状态，包含本轮已经完成的点位。

```bash
bash ~/go2-detail/navigation/routes.sh save farm_route1
```

默认文件是 `~/.local/share/go2-navigation/routes/farm_route1.json`，保存在机器狗宿主机上；退出容器、重启导航或关机不会删除它。同名保存会拒绝覆盖，确需更新时使用 `save farm_route1 --overwrite`。

## 出发前检查、原位重设被挡点

导航运行时，在另一终端检查当前队列，输出每个执行中/待执行点的编号、坐标，
以及该点在静态 `/map` 中是空闲、障碍、未知还是地图范围外：

```bash
bash ~/go2-detail/navigation/routes.sh check
```

程序在出发前检查整条待执行队列，不再走到首点后才发现后面某个目标落在
静态障碍格。空闲目标格不代表路径一定可达；行走期间原有定位、实时代价地图、
足迹碰撞检查继续有效。检查命令只读，不会启用、暂停或取消任务。

例如第 2 点被挡，路线已经暂停时可以仅重设此点，保留其他点位：

```bash
bash ~/go2-detail/navigation/routes.sh replace 2
```

收到“重设模式”成功提示后，在 Foxglove 的导航视图设置一次新的空闲目标。
这次点击替换原来的第 2 点，不会追加到队尾；如果新位置仍不可用，会提示重选。
不能在正在执行的任务中更改目标，`replace` 本身也不会启用运动。
替换后再次 `check`，确认全部点位正确，再点击“重试暂停路线”恢复；尚未开始
的路线点击“启用运动”。手动停止、取消或重新设置初始位姿会撤销待重设操作。

修改的是当前队列；如需下次继续使用修改后的点位，重新保存：

```bash
bash ~/go2-detail/navigation/routes.sh save farm_route1 --overwrite
```

下次先启动**原来同一地图版本**的导航，另开终端加载路线：

```bash
bash ~/go2-detail/navigation/start.sh farm
```

```bash
bash ~/go2-detail/navigation/routes.sh load farm_route1
```

`load` 只恢复点位队列；定位有效后，点击 Foxglove“启用运动”。若明确要通过命令加载并请求开始，可将 `load` 换成 `run`。`run` 的回执只代表已请求，实际执行仍经过定位、Nav2、运动授权和碰撞检查。

查看已存路线或点位内容无需启动导航：

```bash
bash ~/go2-detail/navigation/routes.sh list
bash ~/go2-detail/navigation/routes.sh show farm_route1
```

路线保存地图几何指纹，不能误加载到另一张地图或重新建图后的不同版本。`farm`、`farm_route1` 只是示例名称；同一入口适用于全部地图。已有队列或任务时，加载会拒绝替换；先完成当前路线，或明确点击“停止运动＋取消导航”，再加载。旧地图版本仍可将其完整目录作为 `start.sh` 参数启动。

## 栅格算法更新后的路线重新校验

如果仅导航栅格处理算法更新、原始定位 PCD 和坐标系完全没变，原路线仍会因
地图指纹变化而拒绝直接加载。可以在当前地图导航已启动后明确重新校验，
另存成新路线名：

```bash
bash ~/go2-detail/navigation/routes.sh revalidate farm_route1 --as farm_route1_checked
bash ~/go2-detail/navigation/routes.sh load farm_route1_checked
```

重新校验会寻找与旧路线指纹完全一致的旧缓存，确认新旧 PCD 和坐标参数一致，
并逐点检查当前地图的目标格。全部通过才创建新文件；原文件、点位位置、朝向、
顺序均保留，不会自动修正目标、加载队列或启用运动。旧缓存缺失、新旧几何不同
或任一点仍不可用都会拒绝，请在当前地图重新设置相应点位。此操作使用本机
导航镜像启动无网络、地图和代码均只读的临时校验容器，不连接机器人控制接口。
大地图或历史缓存较多时会逐个输出校验进度。

以上命令不访问 GitHub，也不重新编译。需要代码更新时才单独执行 `git pull`。超时表示回执未确认，尤其执行 `run` 后，应先查看队列和 `/operator/status`，不要仅凭超时直接重复请求。
