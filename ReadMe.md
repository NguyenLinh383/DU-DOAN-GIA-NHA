# **HOUSE'S PRICE PREDICTION**

# I. THÔNG TIN CHUNG

## 1. THÔNG TIN NHÓM
### 1.1. Thông tin thành viên trong nhóm
|**STT**|**MSSV**|**Họ và tên**|**Email**|
|---|-----------------------|--------|---------------------|
|01|20120328|Hoàng Đức Nhật Minh|20120328@student.hcmus.edu.vn|
|02|20120224|Trần Thị Mỹ Trinh|20120224@student.hcmus.edu.vn|
|03|20120210|Trần Thị Kim Tiến|20120210@student.hcmus.edu.vn|
|04|20120307|Phạm Gia Khiêm|20120307@student.hcmus.edu.vn|
|05|20120231|Phan Huy Trường|20120231@student.hcmus.edu.vn|
|06|20120578|Phạm Quốc Thái|20120578@student.hcmus.edu.vn|

### 1.2. Phân chia công việc
#### **Giai đoạn 1**
Trước khi thực hiện chọn ra mô hình tốt nhất, thì chúng em đã chia thành 2 nhóm để cùng thực hiện tiền xử lý, lựa chọn nhiều mô hình khác nhau theo cách khác nhau:
- Nhóm 1, bao gồm các thành viên:
  - Hoàng Đức Nhật Minh
  - Trần Thị Mỹ Trinh
  - Phạm Quốc Thái
- Nhóm 2, bao gồm các thành viên:
  - Phạm Gia Khiêm
  - Trần Thị Kim Tiến
  - Phan Huy Trường

|**Tên công việc**|**Nội dung chi tiết**|**NHÓM 1**|**NHÓM 2**|
|-------------------|-------------------------|--------------------------------|---------------------|
|Phân tích dữ liệu||Hoàng Đức Nhật Minh|Phan Huy Trường|
|Tiền xử lý dữ liệu||Hoàng Đức Nhật Minh <br /> Trần Thị Mỹ Trinh|Phạm Gia Khiêm <br /> Phan Huy Trường|
|Chọn mô hình| 1. Decision tree <br /> 2. Linear regression <br /> 3. Rigde regression <br /> 4. Lasso regression <br /> 5. XGBoost <br /> 6. LightGBM <br /> 7. Bagging <br /> 8. Stacking| <br /> <br /> Phạm Quốc Thái <br /> Trần Thị Mỹ Trinh <br /> Phạm Quốc Thái <br /> Hoàng Đức Nhật Minh <br /> Hoàng Đức Nhật Minh <br /> <br />|Phạm Gia Khiêm <br /> Phạm Gia Khiêm <br /> Phan Huy Trường <br /> Phan Huy Trường<br />Trần Thị Kim Tiến <br /> Trần Thị Kim Tiến <br /> <br /> Phạm Gia Khiêm <br />|

#### **Giai đoạn 2**
Sau giai đoạn 1, nhóm đã họp bàn và chọn ra phương pháp tiền xử lý hiệu quả nhất là của *NHÓM 1* và lựa chọn mô hình huấn luyện cho dữ liệu phù hợp nhất đó là các mô hình:
- Ridge regression
- Lasso regression
- XGBoost
- LightGBM
- Bagging 
- Stacking

Sau khi lựa chọn mô hình, thì nhóm bắt đầu phân chia tiếp công việc như sau:
|**Tên công việc**|**Thành viên đảm nhiệm**|**Đánh giá**|
|-------------------|-------------------------|---------------------|
|Tìm hyperparameter cho các mô hình|Phạm Gia Khiêm <br /> Hoàng Đức Nhật Minh|100%|
|Tìm hiểu Ridge <br /> Tìm hiểu Lasso|Phạm Quốc Thái <br /> Phan Huy Trường|100%|
|Tìm hiểu XGBoost <br /> Tìm hiểu LightGBM|Trần Thị Kim Tiến|100%|
|Làm slide thuyết trình|Trần Thị Mỹ Trinh <br /> Hoàng Đức Nhật Minh <br />Trần Thị Kim Tiến <br /> Phạm Gia Khiêm <br /> Phan Huy Trường <br /> Phạm Quốc Thái|100%|
|Thuyết trình|Phạm Quốc Thái <br /> Trần Thị Kim Tiến|100%|

## 2. MÔ TẢ DỮ LIỆU - DATASET DESCRIPTION
### 2.1. Thông tin chung:
Bộ dữ liệu này được lấy từ một cuộc thi của kaggle nhằm dự đoán giá nhà từ các thông tin thu thập được như là diện tích nhà, số phòng, tầng hầm,...  
- Tên bộ dữ liệu:  House Prices - Advanced Regression Techniques
- Đường dẫn của cuộc thi: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
### 2.2. Mô tả file - File descriptions
- train.csv - bộ dữ liệu huấn luyện
- test.csv - bộ dữ liệu kiểm thử
- data_description.txt - Toàn bộ những mô tả về thông tin của từng cột thuộc tính của dữ liệu
- sample_submission.csv - mẩu chuẩn để nộp bài lên kaggle
### 2.3. Trường dữ liệu - Data fields
Giới thiệu sơ qua một vài trường dữ liệu của dữ liệu:
- `SalePrice` - Giá bán của ngôi nhà.
- `OverallQual` - Chất lượng toàn bộ nội thất trong nhà.
- `GrLivArea` - Tổng diện tích ngôi nhà từ mặt đất trở lên.
- `YearBuilt` - số năm được sử dụng.
- `YearRemodAdd` - Ngày sửa sang lại.
- `TotalBsmtSF` - Tổng diện tích tầng hầm.
- `FullBath` - Tổng số phòng tắm của ngôi nhà.

# II. QUÁ TRÌNH THỰC HIỆN ĐỒ ÁN

## 1. Khám phá dữ liệu
### 1.1. Sơ bộ về dữ liệu
- Kích thước tập dữ liệu:
  - Tập train: (1460, 80)
  - Tập test: (1459, 79)
- Cả tập train và tập test có 79 đặc trưng (cột) độc lập trong đó:
  - Có 37 cột là thuộc loại Numerical (Trong đó có cột output - `SalePrice`).
  - Có 43 cột là thuộc loại Categorical. 
- Xét dữ liệu thiếu:
  - Tập train: Có 19 cột bị thiếu dữ liệu.
  - Tập test: Có 33 cột bị thiếu dữ liệu.
### 1.2. Phân tích biến Numerical
#### 1.2.1. Phân tích độ tương quan
- Xét độ tương quan của các biến numerical với SalePrice, ta thấy có một vài biến có độ tương quan > 0.5.
- Tuy nhiên, có một vài cặp independent variable tương quan mạnh với nhau: 
  - `TotalBsmtSF` - `1stFlrSF`: Diện tích của basement area ~ Diện tích của tầng 1.
  - `YearBuilt` - `GarageYrBlt`: Ngày nhà xây dựng ~ số năm mà cái Garage đã được sử dụng từ ngày xây dựng
  - `GrLivArea` - `TotRmsAbvGrd`: Tổng diện tích ngôi nhà từ mặt đất trở lên ~ Tổng số phòng trong nhà ngoại trừ phòng tắm từ tầng mặt đất trở lên.
  - `GarageCars` - `GarageArea`: Số lượng chiếc xe có thể để trong garage ~ kích thước garage 
- Những biến này, ta sẽ bỏ đi 1 trong 2 vì biết được 1 biến ta có thể suy ra giá trị của biến còn lại, và tất nhiên ta sẽ giữ lại những biến nào có tương quan mạnh hơn với dependent variable. Đây gọi là vấn đề **multicolinearity**

#### 1.2.2. Phân tích độ phân phối 

### 1.3. Phân tích biến Categorical

## 2. Tiền xữ lý dữ liệu
### 2.1. Impute: điền dữ liệu thiếu
Ta có 3 nhóm thuộc tính khác nhau về loại thuộc tính và giá trị định nghĩa của từng thuộc tính.
- Nhóm 1: Biến categorical với NA nghĩa là `None`, có nghĩa là không có. Cách xử lý: Thay thế những những vị trí có giá trị NA thành None.
- Nhóm 2: Ngược lại với nhóm 1, biến categorical với NA không có nghĩa là `None`, có nghĩa là có. Thay thế những vị trí có giá trị NA thành mode (giá trị phổ biến) của cột thuộc tính đó. 
- Nhóm 3: Biến numerical với NA nghĩa là `0`, giá trị 0. Thay thế những vị trí có giá trị NA thành 0.
- Trường hợp đặt biệt Cột thuộc tính LotFrontage ta sẽ thay th61 những vị trí có giá trị NA thành mean (giá trị trung vị) của cột thuộc tính LotFrontage.

### 2.2. Loại bỏ nhiễu (outlier)
Ta kiểm tra outlier ở các thuộc tính: `GrLiveArea`, `TotalBsmtSF`, `Street`, `LandContour`. Ta có thể nhận xét sơ lược:
- GrLiveArea: Có 2 dữ liệu nhiễu.
- TotalBsmtSF và Street: Chưa thể chắc chắn được là đó có phải là dữ liệu nhiễu hay không.
- LandContour: Những điểm dữ liệu được cho là outlier thực chất là sai vì những điểm đó còn bị chi phối bởi các thuộc tính khác.

### 2.3. Feature Engineering
#### 2.3.1. Tạo biến mới
#### 2.3.2. Label Encoding
Mã hóa cho ordinal categorical variable
#### 2.3.3. Biến đổi Numerical Variable sang Categorical Variable
- `YrSold` có thể tác động lớn tới giá (ví dụ năm đó khủng hoảng kinh tế, chiến tranh,...giá nhà có thể thấp). Do đó, ta chuyển chúng sang dạng categorical.
- `MoSold`, `MSSubClass` là những biến numeric nhưng không có ý nghĩa gì về mặt thứ tự nên ta chuyển nó sang dạng categorical.
#### 2.3.4. Skewness và chuẩn hoá dữ liệu:
- Chuẩn hoá cột Target: Một trong những cách để normalize right-skewd data là dùng log transformation vì giá trị lớn sẽ bị kéo về giữa. Tuy nhiên log(0) = nan, nên chúng ta sẽ dùng log(1 + x) để chuẩn hoá giá trị target (SalePrice).
- Bước tiếp theo, kiểm tra skewness của các numerical variable còn lại và dùng log transformation để chuẩn hóa.
  - Nếu skewness <= -1 or >= 1, thì phân phối bị highly skewed.
  - Nếu skewness nằm giữa -1 và -0.5 hoặc 0.5 và 1 thì phân phối bị moderately skewed.
  - Nếu skewness nằm giữa -0.5 và 0.5 thì phân phối approximately symmetric.
#### 2.3.5. Normalize (chuẩn hóa dữ liệu) - Feature scaling
Ngoại trừ Decision Tree và Random Forest, các thuật toán còn lại trong machine learning nên được standardize để mô hình nhanh hội tụ và ổn định.

#### 2.3.6. One-hot encoding
Chuyển các dữ liệu từ Categorical sang Numerical.

### **Kết quả của quá trình tiền xử lý dữ liệu:** 
Sau khi tiền xử lý dữ liệu, chúng ta có tập dữ liệu với kích thước mới:
- Tập train: (1460, 229)
- Tập test: (1459, 228)

Sau đó, ta sẽ ghi tập dữ liệu train và tập test mới này ra ngoài file khác và tiến đến quá trình chọn mô hình.

## 3. Xây dựng mô hình
Lựa chọn mô hình: Có rất nhiều mô hình để thực hiện cho bộ dữ liệu này, ví dụ: 
- Các mô hình cơ bản, đơn giản: Linear Regression, Random Forests Regressor
- Các mô hình phát triển hơn: Ridge, Lasso, ...
- Các mô hình nâng cao: GBoost, XGBoost, Light GBM, ...

### 3.1. Mô hình Ridge
### 3.2. Mô hình Lasso 
### 3.3. Mô hình XGBoost
### 3.4. Mô hình LightGBM
### 3.5. Mô hình Ensemble
Sử dụng các thông số tốt nhất của 4 mô hình Ridge, Lasso, XGBoost và LightGBM và kết hợp 4 mô hình này lại với nhau.

Có 2 cách kết hợp mô hình
#### 3.5.1. Bagging
- Dùng bagging, cho 4 mô hình huấn luyện song song với nhau và ta sẽ dùng trung bình cộng các kết quả của 4 mô hình. 
- Kết quả thu được của hô mình Bagging là rất tốt (rmse = 0.11058).

#### 3.5.2. Stacking
- Dùng stacking để kết hợp các mô hình với base models là (Rigde, Lassso, XGBoost, LightGBM) và meta model là Ridge. 
- Kết quả của mô hình này rất tốt, chênh lệch không nhiều so với Bagging (rmse = 0.1115).

## 4. Đánh giá mô hình
Trong model evaluation, thường thì ta sẽ chia toàn bộ data làm 2 tập train và test. Tuy nhiên, đối với bộ dữ liệu này khá nhỏ, nên nếu chia như vậy, thì có lẽ mô hình sẽ dễ bị overfitting. Do đó, ta dùng `cross-validation` là K-Fold với k = 5 để huấn luyện mô hình.

Đề bài yêu cầu đánh giá mô hình bằng độ đo RMSE, tuy nhiên do target variable đã được biến đổi qua log(1 + y), nên MSE cho log(1 + y) là MSLE - Mean Squared Logarithmic Error.

# III. KẾT QUẢ ĐẠT ĐƯỢC
## 1. Kết quả đạt được trên kaggle
![image](https://github.com/hoangducnhatminh/predict-house-price-advanced/assets/94270107/12bb75de-62d0-40b8-a4c4-12b88cf84fe2)

![image](https://github.com/hoangducnhatminh/predict-house-price-advanced/assets/94270107/cce856af-0757-4de6-878d-334ad3cad54a)
Với số điểm RMSE đạt được là 0.11973, thì thứ hạng mà nhóm đạt được là 228 / 4607 teams (chưa loại trừ các bài nộp gian lận).

## 2. Những khó khăn và hạn chế trong quá trình thực hiện đồ án
