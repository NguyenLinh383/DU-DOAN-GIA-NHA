# PHỤ LỤC

## 1. Thông tin nhóm

| STT | MSSV | Họ và tên | Email |
|-----|------|-----------|-------|
| 1 | 20120328 | Hoàng Đức Nhật Minh | 20120328@student.hcmus.edu.vn |
| 2 | 20120224 | Trần Thị Mỹ Trinh | 20120224@student.hcmus.edu.vn |
| 3 | 20120210 | Trần Thị Kim Tiến | 20120210@student.hcmus.edu.vn |
| 4 | 20120307 | Phạm Gia Khiêm | 20120307@student.hcmus.edu.vn |
| 5 | 20120231 | Phan Huy Trường | 20120231@student.hcmus.edu.vn |

## 2. Mô tả dữ liệu

### 2.1. Các trường dữ liệu chính
- **SalePrice**: Giá bán nhà (biến mục tiêu)
- **OverallQual**: Chất lượng tổng thể
- **GrLivArea**: Diện tích sinh hoạt
- **GarageCars**: Số lượng ô tô chứa được trong garage
- **TotalBsmtSF**: Tổng diện tích tầng hầm
- **FullBath**: Số phòng tắm đầy đủ
- **YearBuilt**: Năm xây dựng
- **YearRemodAdd**: Năm cải tạo gần nhất

### 2.2. Thống kê mô tả
```python
# Mã xem thống kê mô tả
train_data.describe()
```

Kết quả (một phần):
```
       LotArea  OverallQual  OverallCond    YearBuilt  YearRemodAdd   TotalBsmtSF  \
count  1460.00   1460.000000  1460.000000   1460.000000   1460.000000   1460.000000   
mean   10517.00      6.099315     5.575342   1971.267808   1984.865753   1057.429452   
std     9981.26      1.382997     1.112799     30.202904     20.645407    438.705324   
min     1300.00      1.000000     1.000000   1872.000000   1950.000000      0.000000   
25%     7554.00      5.000000     5.000000   1954.000000   1967.000000    795.750000   
50%     9478.50      6.000000     5.000000   1973.000000   1994.000000    992.000000   
75%    11602.00      7.000000     6.000000   2000.000000   2004.000000   1298.250000   
max  215245.00     10.000000     9.000000   2010.000000   2010.000000   6110.000000   
```

## 3. Các mô hình chính

### 3.1. Mô hình Ridge Regression
```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge = Ridge()
parameters = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(X_train, y_train)
```

### 3.2. Mô hình Lasso Regression
```python
from sklearn.linear_model import Lasso

lasso = Lasso()
parameters = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(X_train, y_train)
```

### 3.3. Mô hình XGBoost
```python
import xgboost as xgb

xgb_model = xgb.XGBRegressor()
parameters = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}
xgb_regressor = GridSearchCV(xgb_model, parameters, scoring='neg_mean_squared_error', cv=5)
xgb_regressor.fit(X_train, y_train)
```

## 4. Kết quả đánh giá mô hình

### 4.1. Bảng so sánh hiệu suất

| Mô hình | RMSE (Train) | RMSE (Test) | R² (Train) | R² (Test) |
|---------|-------------|-------------|------------|------------|
| Ridge | 0.12 | 0.13 | 0.89 | 0.88 |
| Lasso | 0.11 | 0.12 | 0.90 | 0.89 |
| XGBoost | 0.08 | 0.10 | 0.94 | 0.92 |

### 4.2. Biểu đồ so sánh dự đoán và thực tế

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Giá thực tế')
plt.ylabel('Giá dự đoán')
plt.title('So sánh giá dự đoán và giá thực tế')
plt.show()
```

## 5. Feature Importance từ XGBoost

```python
importances = xgb_regressor.best_estimator_.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices])
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.tight_layout()
plt.show()
```

## 6. Kết luận

- Mô hình XGBoost cho kết quả tốt nhất với R² = 0.92 trên tập test
- Các yếu tố ảnh hưởng nhiều nhất đến giá nhà:
  1. Chất lượng tổng thể (OverallQual)
  2. Diện tích sinh hoạt (GrLivArea)
  3. Tổng diện tích tầng hầm (TotalBsmtSF)
  4. Số lượng xe chứa được trong garage (GarageCars)
  5. Năm xây dựng (YearBuilt)

## 7. Tài liệu tham khảo
1. Tài liệu chính thức của scikit-learn: https://scikit-learn.org/
2. Tài liệu XGBoost: https://xgboost.readthedocs.io/
3. Tài liệu pandas: https://pandas.pydata.org/
