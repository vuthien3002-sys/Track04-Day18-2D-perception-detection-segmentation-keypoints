# Bài tập về nhà 3 — Đo cả pipeline: YOLO26n ONNX trên CPU

Link notebook đã chạy: https://github.com/vuthien3002-sys/Track04-Day18-2D-perception-detection-segmentation-keypoints/blob/main/submission/bai_tap_ve_nha_onnx.ipynb

Notebook nằm ngay trong thư mục này: [`bai_tap_ve_nha_onnx.ipynb`](bai_tap_ve_nha_onnx.ipynb) (đã chạy trên Colab, còn nguyên output).
Notebook bài làm chính: https://github.com/vuthien3002-sys/Track04-Day18-2D-perception-detection-segmentation-keypoints/blob/main/lab_2d_perception_student.ipynb
Môi trường: Intel Xeon CPU @ 2.00GHz, 2 luồng · `ultralytics 8.4.171` · `onnxruntime 1.30.0` (CPUExecutionProvider) ·
`onnx 1.23.1`, opset 18. Mọi lệnh export và predict đều đặt `device="cpu"`.

## Export hai head

| Head | Lệnh | Input → output của graph | Cờ `end2end` trong metadata |
|---|---|---|---|
| one-to-many + NMS | `model.export(format="onnx")` (`nms=None`) | `[1, 3, 640, 640]` → `[1, 84, 8400]` | `False` → NMS chạy ngoài graph |
| one-to-one NMS-free | `model.export(format="onnx", nms=False)` | `[1, 3, 640, 640]` → `[1, 300, 6]` | `True` → chỉ lọc theo conf |

Cả hai file nặng khoảng 9.9 MB và đều đếm đúng **4 người, 1 xe buýt** trên `bus.jpg` ở conf 0.25.

## Số dự đoán vượt ngưỡng conf

"Cảnh đông" là `crowd.jpg`, ghép 2×2 từ `bus.jpg`. Với head one-to-many, đây chính là số ứng viên đưa vào NMS.

| Ảnh | one-to-many, conf 0.25 | one-to-many, conf 0.001 | one-to-one, conf 0.25 | one-to-one, conf 0.001 |
|---|---:|---:|---:|---:|
| bus.jpg | 48 | 620 | 5 | 177 |
| crowd.jpg | 173 | 1379 | 18 | 300 (chạm trần top-300) |

## Latency trên CPU (ms/ảnh, trung bình 30 lần)

| Ảnh | Head | conf | preprocess | inference | postprocess | tổng | số box |
|---|---|---|---:|---:|---:|---:|---:|
| bus.jpg | one-to-many + NMS | 0.25 | 4.87 | 76.41 | 1.37 | 82.65 | 5 |
| bus.jpg | one-to-many + NMS | 0.001 | 4.60 | 75.59 | 2.06 | 82.25 | 186 |
| bus.jpg | one-to-one NMS-free | 0.25 | 5.00 | 81.68 | 0.44 | 87.12 | 5 |
| bus.jpg | one-to-one NMS-free | 0.001 | 6.50 | 103.15 | 0.65 | 110.30 | 177 |
| crowd.jpg | one-to-many + NMS | 0.25 | 5.45 | 83.12 | 1.56 | 90.13 | 18 |
| crowd.jpg | one-to-many + NMS | 0.001 | 4.91 | 76.45 | 4.69 | 86.04 | 300 |
| crowd.jpg | one-to-one NMS-free | 0.25 | 6.01 | 93.57 | 0.46 | 100.04 | 18 |
| crowd.jpg | one-to-one NMS-free | 0.001 | 5.50 | 87.90 | 0.51 | 93.90 | 300 |

Đối chiếu với GPU T4 trong notebook chính (PyTorch, `bus.jpg` letterbox 640×480): inference 9.98–13.28 ms;
postprocess của one-to-many + NMS là 1.28 → 1.65 ms, của one-to-one là 0.59 / 0.56 ms.

## Nhận xét

1. **Chi phí NMS tăng theo số ứng viên, head one-to-one thì không.** Postprocess của one-to-many tăng từ 1.37 ms
   (48 ứng viên) lên 2.06 ms (620), rồi 4.69 ms với cảnh đông ở conf 0.001 (1379 ứng viên), tức gấp 3.4 lần. Postprocess
   của one-to-one giữ khoảng 0.44–0.65 ms ở mọi trường hợp. Đây đúng là lợi ích "khi conf thấp, khi cảnh đông" của
   NMS-free: thời gian hậu xử lý ổn định, không phụ thuộc nội dung ảnh.
2. **Trên CPU này, inference chiếm hơn 90% thời gian.** Inference mất 75–103 ms, chậm khoảng 7 lần so với T4, trong khi
   phần NMS tiết kiệm được chỉ khoảng 1–4 ms. Vì vậy tổng thời gian của one-to-one **không nhanh hơn** trong lần đo này
   (87–110 ms so với 82–90 ms).
3. **Phần chênh inference chủ yếu là nhiễu đo.** Cùng một graph one-to-one, inference là 81.68 ms ở conf 0.25 nhưng
   103.15 ms ở conf 0.001, dù conf không hề tác động lên graph. CPU 2 luồng dùng chung trên Colab dao động cỡ 10–25 ms
   giữa các lần đo, lớn hơn cả phần NMS tiết kiệm được. Graph one-to-one cũng gánh thêm phép TopK (k = 300) bên trong,
   tức một phần công "chọn box" được dời từ hậu xử lý vào inference.
4. **Khi nào NMS-free thật sự đáng giá.** Ở đây NMS chạy bằng torchvision (C++) ngay trên CPU nên khá rẻ. Lợi ích lớn
   hơn khi: (a) triển khai trên NPU/edge mà NMS không biên dịch được vào graph, phải chép tensor về CPU để xử lý;
   (b) cần latency ổn định, vì NMS có thời gian phụ thuộc số ứng viên; (c) cảnh rất đông hoặc conf thấp, nơi số ứng viên
   lên tới hàng nghìn. Head one-to-one cũng làm graph tự chứa (output đã là box cuối), nên dễ export và tích hợp hơn.

**Hạn chế:** mỗi cấu hình chỉ đo 30 lần trên một máy ảo dùng chung, nên các chênh lệch inference dưới 10–20 ms chưa đáng
tin. Muốn so sánh chặt hơn, nên đo nhiều lần hơn, lấy median, cố định số luồng của onnxruntime, và đo thêm trên thiết bị
đích (ví dụ OpenVINO trên Intel, hoặc NPU).
