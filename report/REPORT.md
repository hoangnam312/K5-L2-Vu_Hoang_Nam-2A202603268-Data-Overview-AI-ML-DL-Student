# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 6/11/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không 

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: 
{
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "image_classification",
    "taxonomy_name": "ImageNet-1K",
    "model_file": "yolo11n-cls.pt",
    "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "ultralytics_version": "8.4.145",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
  }
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? 
Nó nói về **một vật thể riêng** bởi vì trong file đang đề cập đến loại xe cụ thể là xe **cab**
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Nó đi kèm taxonomy
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- taxonomy_name để cho biết hệ thống đang phân loại bằng taxonomy nào
- class_id dùng để biết mã id của vật thể trong taxonomy
- class_name là tên của vật thể
Việc ghi cả ba là điều cần thiết vì cùng một bức ảnh nếu dùng các taxonomy khác nhau sẽ ra các tên khác nhau, tương tự cũng như khi có một record chúng ta cần các thông tin này để biết model này dùng taxonomy nào và id của nó là gì để có thể dễ dàng so sánh cũng như sử dụng với các record khác vì phải cùng một taxonomy mới có thể sử dụng chính xác và ổn định được.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần nêu rõ các loại vật/ chủ đề cần gán nhán. Ví dụ cùng một bức ảnh giữa ngã tư phố phường Hà Nội. Nếu:
- hệ thống yêu cầu nhận diện thông tin người thì gán nhãn các đối tượng là con người
- hệ thống nhận biết biển số xe thì gãn nhãn biển số
- hệ thống đếm và phân loạn phương tiện thì gán nhãn phương tiện
Tùy theo output của bài toán mong muốn thì chúng ta phải nêu rõ loại vật/ chủ đề cần gán nhãn để tránh tình trạng mập mờ, mơ hồ dữ liệu (ambiguity)
6. Vì sao không được lấy `score` của model làm ground truth?
- score chỉ là độ tự tin về mặt toán học của mô hình tại thời điểm hiện tại dựa trên những gì nó đã được học. Nó có thể đúng, có thể sai, có thể không chính xác
- ground truth là đáp án chuẩn xác do con người xác nhận, nó phải đc lấy từ con người chứ không phải model
- Tránh lỗi lũy tiến: Nếu lấy score làm nhãn chuẩn, ta có thể khiến cho mô hình học cái sai, mập mờ, định kiến của mô hình cũ

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm 
polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
{
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "instance_segmentation",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n-seg.pt",
    "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "instance_id": "traffic-001",
    "class_id": 5,
    "class_name": "bus",
    "score": 0.925745,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      95.3,
      188.72,
      224.08,
      319.96
    ],
    "polygon_point_count": 120,
    "polygon_xy": [
      [
        148.0,
        189.0
      ],
      [
        147.0,
        190.0
      ],
      [
        145.0,
        190.0
      ],
      [
        143.0,
        192.0
      ],
      ...
    ]
}
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Polygon cho ta biết về hình dáng của vật thể, box (	Bounding Box ) cho ta khung hình chữ nhật bao quanh về vật thể, polygon cho ta biết về hình dáng thực sự của vật thể (ôm sát viền)
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- instance_id: id của vật thể trong ảnh. Ví dụ có 2 chiếc xe bus thì một cái sẽ có id 1, cái còn lại sẽ có id 2
- class_id: vật thể này là gì: bus, cab, .... Đây chỉ là ví dụ về cách diễn đạt còn trong thực tế mã sẽ có dạng số
- tracking_id: mã vật thể trong suốt một đoạn video (nhiều khung hình). Ví dụ một chiếc xe bus đi trên một con đường thì mã tracking_id của chiếc xe này trên cam đầu đường, cuối đường luôn giống nhau
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- Quy tắc ôm sát (Tightness Rule): Các điểm đa giác phải được chấm sát nhất có thể vào đường biên thực tế của vật thể. Không được chấm quá rộng ra ngoài (lọt pixel nền) hoặc cắt lẹm vào bên trong vật thể.
- Số lượng điểm vừa đủ (Density Rule):
  - Với những đường viền thẳng (như cạnh bàn, cạnh lò nướng): Chỉ cần chấm ít điểm ở các góc.
  - Với những đường cong, tròn (như cái bát, quả táo, con người): Phải chấm nhiều điểm dày hơn để thể hiện độ cong mượt mà của vật thể.
- Quy tắc xử lý vật thể bị che khuất (Occlusion Rule):
  - Nếu vật thể bị che khuất một phần bởi vật khác (ví dụ: người đứng sau cái bàn bếp), ta chỉ vẽ mask cho phần nhìn thấy được của người đó, tránh vẽ đè lên diện tích của cái bàn bếp.
  - Nếu vật thể bị cắt làm hai phần rời rạc do một vật thể khác chắn ngang ở giữa, guideline thường yêu cầu vẽ 2 polygon độc lập nhưng gán chung một instance_id (hoặc theo quy định cụ thể của từng dự án).
- Quy tắc xử lý biên mờ/Shadow (Blur & Shadow Rule):
  - Không gán nhãn phần bóng đổ (shadow) của vật thể trừ khi có yêu cầu đặc biệt.
  - Đối với các viền bị mờ do chuyển động (motion blur) hoặc thiếu sáng, annotator phải ước lượng ranh giới trung bình dựa trên kinh nghiệm cảm quan trực giác thay vì bỏ qua.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Hai vật chạm nhau / che khuất (Occlusion): Khu vực các nồi chảo bằng đồng/kim loại treo trên giá phía trên bàn bếp. Chúng treo sát, đè lên và che khuất nhau
Dùng theo thứ tự sau:
- Đọc kĩ guideline
- Nếu guideline không có thì dùng escalation (hỏi lead)

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn duy nhất cho toàn bộ ảnh (thuộc danh mục ImageNet-1K). |  | Xem toàn bộ bức ảnh và gán 1 nhãn lớp phù hợp nhất đại diện cho chủ thể hoặc ngữ cảnh chính của bức ảnh theo danh mục (taxonomy) có sẵn. | Kiểm tra xem nhãn được gán có đúng không, có bị nhầm lẫn giữa các lớp tương tự (ví dụ: cab với minibus hay police_van) không |
| Phát hiện vật thể | Tập hợp các hình hộp chữ nhật (bounding boxes) bao quanh từng vật thể, mỗi hộp đi kèm với 1 nhãn lớp (COCO-80) và tọa độ [x_min, y_min, x_max, y_max] tính theo pixel. | Ở ảnh kitchen với threshold 0.35, mô hình nhận diện thiếu nhiều chiếc muỗng (spoon) treo trên tường hoặc các chai lọ nhỏ trên bàn. | Vẽ các hộp giới hạn (bounding boxes) khít nhất có thể quanh từng vật thể mục tiêu, gán đúng nhãn lớp tương ứng cho mỗi hộp. | Kiểm tra xem các hộp có bị vẽ quá rộng, quá hẹp, hoặc bị sót vật thể không; xác minh các trường hợp vật thể chồng chéo lên nhau đã được gán nhãn chính xác chưa. |
| Instance segmentation | Tập hợp các đa giác (polygons) gồm chuỗi tọa độ [x, y] bao quanh khít đường viền của từng thực thể vật thể riêng biệt, kèm theo nhãn lớp tương ứng. | Ở ảnh kitchen, khu vực chiếc bàn gỗ (dining table với score 0.58) bị đa giác tô đè lên cả các đĩa thức ăn và cây lăn bột đặt phía trên | Vẽ tỉ mỉ đường viền (polygon) bám khít theo rìa/biên của từng đối tượng, phân biệt rõ các cá thể (instances) riêng biệt dù chúng cùng một lớp và đứng sát nhau. | Kiểm tra độ mịn và độ chính xác của đường viền đa giác (không cắt lẹm vào vật thể hoặc chừa quá nhiều khoảng trống); rà soát xem các phần bị che khuất có bị phân mảnh sai hay không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ sử dụng các hình ảnh COCO công khai có sẵn được cung cấp trong bài lab và tuyệt đối không tải lên dữ liệu cá nhân hay dữ liệu nội bộ của doanh nghiệp.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Escalation (Xử lý khi gặp lỗi): Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi hoặc các ca gán nhãn/quyết định nhãn khó, bạn cần dừng lại ngay và báo cáo cho Lab Coach / Mentor để được hỗ trợ giải quyết, tuyệt đối không tự ý bịa nhãn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
