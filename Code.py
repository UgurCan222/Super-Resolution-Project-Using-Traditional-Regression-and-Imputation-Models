import numpy as np
import cv2
import xgboost as xgb

input_path = r'C:\github\Super-Resolution-Project-Using-Traditional-Regression-and-Imputation-Models\input.bmp'
output_path = r'C:\github\Super-Resolution-Project-Using-Traditional-Regression-and-Imputation-Models\output.bmp'

image = cv2.imread(input_path, cv2.IMREAD_COLOR)
rows, cols, channels = image.shape

params = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100
}

new_cols = 2 * cols - 1
new_image = np.zeros((rows, new_cols, channels), dtype=np.uint8)

for row in range(rows):
    new_col_idx = 0
    for col in range(cols - 1):
        for ch in range(channels):
            new_image[row, new_col_idx, ch] = image[row, col, ch]
        
            
            X_train = np.array([[image[row, col, ch], image[row, col + 1, ch]]])
            y_train = np.array([image[row, col + 1, ch]])
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            
            X_test = np.array([[image[row, col, ch], image[row, col + 1, ch]]])
            predicted_value = model.predict(X_test)[0]
            new_image[row, new_col_idx + 1, ch] = np.clip(predicted_value, 0, 255).astype(np.uint8)

        new_col_idx += 2
    
    
    new_image[row, new_col_idx, :] = image[row, cols - 1, :]


new_rows = 2 * rows - 1
final_image = np.zeros((new_rows, new_cols, channels), dtype=np.uint8)

for col in range(new_cols):
    new_row_idx = 0
    for row in range(rows - 1):
        final_image[new_row_idx, :, :] = new_image[row, :, :]
        new_row_idx += 1
        
        
        for ch in range(channels):
            X_train = np.array([[new_image[row, col, ch], new_image[row + 1, col, ch]]])
            y_train = np.array([new_image[row + 1, col, ch]])
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            
            X_test = np.array([[new_image[row, col, ch], new_image[row + 1, col, ch]]])
            predicted_value = model.predict(X_test)[0]
            final_image[new_row_idx, col, ch] = np.clip(predicted_value, 0, 255).astype(np.uint8)
    
        new_row_idx += 1

    
    final_image[new_row_idx, :, :] = new_image[rows - 1, :, :]


cv2.imwrite(output_path, final_image)

print(f"DONE")
