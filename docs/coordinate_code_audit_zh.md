# pyLAST 坐标系代码审计（main）

本文结论来自 pyLAST 执行代码和回归测试，不以旧说明或绘图标签作为依据。

审计基线是 `main` 提交 `7a65cf1f12e37e0e437968c37da34b9a7fc4dd08`，并快进纳入其上的 LACT ROOT reader 提交至 `953d07392a542a3362a0ce35f14bcdbe4c347c9a`。

## 天空坐标

`include/Coordinates.hh::SphericalRepresentation::transform_to_cartesian()` 实际计算：

```text
(x, y, z) = (cos az cos alt, -sin az cos alt, sin alt)
```

所以它是 NWU：`+x=North`、`+y=West`、`+z=Up`；`az=0` 指北，正方位角朝东。

## TelescopeFrame

`include/CoordFrames.hh` 构造指向旋转；`src/CoordFrames.cpp` 把天空方向投影成：

```text
pix_x/source_offset_x = -local_direction_x / local_direction_z
pix_y/source_offset_y = -local_direction_y / local_direction_z
```

数值回归 `test/test_coordinates.cpp` 要求：

- 比望远镜指向更高的光源得到 `pix_x > 0`；
- 比望远镜指向更东的光源得到 `pix_y > 0`；
- 西侧光源得到 `pix_y < 0`。

因此 pyLAST 的坐标是天空/source-offset 方向，不是硬件焦平面上反射光斑的原始方向。

## LACT ROOT 读取边界

LACT_sim ROOT 的相机像素 `x_m/y_m` 是原始焦平面 `u/v`：`+u` 指东，`+v` 为 sky-up。镜面成像使天空东西与上下都落在硬件平面的反方向，所以
`root/LactEventSource.cpp::load_telescopes()` 只转换一次：

```text
pyLAST pix_x = -LACT v
pyLAST pix_y = -LACT u
```

`root/test_lact_event_source.cpp` 使用非零 `(u,v)` 样本检查两个交换和符号，防止对称图像掩盖错误。转换不重排 `pixel_id`、PE 或波形数组。

## 绘图

`src/pylast/visualize/visualize.py::plot_camera_image()` 使用：

```text
screen horizontal = pix_y
screen vertical   = pix_x
```

因此屏幕右侧是天空东，屏幕上方是天空向更高高度角；这只是 pyLAST 原生轴顺序，不应在 LACT 专用绘图函数里再次翻转。
