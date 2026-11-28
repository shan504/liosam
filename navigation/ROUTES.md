# 多点路线保存与复用

导航运行时，在 Foxglove 中按顺序设置目标。等航点编号出现后，在另一终端保存；保存不改变运动状态，包含本轮已经完成的点位。

```bash
bash ~/go2-detail/navigation/routes.sh save farm_route1
```

默认文件是 `~/.local/share/go2-navigation/routes/farm_route1.json`，保存在机器狗宿主机上；退出容器、重启导航或关机不会删除它。同名保存会拒绝覆盖，确需更新时使用 `save farm_route1 --overwrite`。

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

以上命令不访问 GitHub，也不重新编译。需要代码更新时才单独执行 `git pull`。超时表示回执未确认，尤其执行 `run` 后，应先查看队列和 `/operator/status`，不要仅凭超时直接重复请求。
