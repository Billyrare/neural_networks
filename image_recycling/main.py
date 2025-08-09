# Лабораторная работа по дисциплине МРЗВИС
# 6 вариант 
# Реализация рециркуляционной нейронной сети с адаптивным коэффициентом обучения и ненормированными весами
# Выполнил студент группы 121731
# Исамиддинов Ботир
#
#
# за основу взят код: https://github.com/Drapegnik/bsuir/tree/master/data-mining/lab1

import numpy as np
from PIL import Image

class RecyclingNetwork:
    def __init__(self, input_size, hidden_size):
        limit = np.sqrt(2./ (input_size + hidden_size))
        self.Wf = np.random.uniform(-limit, limit, (input_size, hidden_size))
        self.Wb = np.random.uniform(-limit, limit, (hidden_size, input_size)) 
        # print("\n WF: ",self.Wf)
        # print("\n WB: ",self.Wb)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.mse_history = []
        
    def forward(self, X):
        H = np.dot(X, self.Wf)
        return H 

    def backward(self, H):
        X_reconstructed = np.dot(H, self.Wb)
        return X_reconstructed

    def calculate_mse(self, error):
        return np.sum(np.square(error))/2

    def calculate_adaptive_learning_rate(self, X, H):
        x_squared_sum = np.sum(np.square(X), axis=1, keepdims=True)
        h_squared_sum = np.sum(np.square(H), axis=1, keepdims=True)
        
        base_lr = 1.0
        decay = 1.0
        
        alpha_forward = base_lr * decay/(np.sum(x_squared_sum) + 1e-8)
        alpha_backward = base_lr * decay/(np.sum(h_squared_sum) + 1e-8)
     
        return float(alpha_forward), float(alpha_backward)

    def train(self, X, epochs=2000, target_mse=1e-5):
        best_error = float('inf')
        best_weights = {'Wf': None, 'Wb': None}
        mse_check_interval = 100
        patience = 10
        no_improvement = 0
        
        
        total_pixels = X.shape[1] // 3  
        r = m = int(np.sqrt(total_pixels))  
        
        print(f"Размерность входного вектора X: {X.shape} (количество блоков × размер блока)")
        print(f"Где {X.shape[0]} = количество блоков ({256//8}x{256//8})")
        print(f"    {X.shape[1]} = размер блока ({r}x{m}x3 канала)")
        
        
        for epoch in range(epochs):
            H = self.forward(X) #x - ?x?x?
            X_reconstructed = self.backward(H)
            error = X - X_reconstructed
            
            if epoch % mse_check_interval == 0:
                current_mse = self.calculate_mse(error)
                self.mse_history.append(current_mse)
                print(f"Эпоха {epoch}, СК: {current_mse:.6f}")
                
                if current_mse < best_error:
                    best_error = current_mse
                    best_weights['Wf'] = self.Wf.copy()
                    best_weights['Wb'] = self.Wb.copy()
                    no_improvement = 0
                else:
                    no_improvement += 1
                
#                if current_mse < target_mse or no_improvement >= patience:
#                    print(f"Stopping at epoch {epoch}")
#                    self.Wf = best_weights['Wf']
#                    self.Wb = best_weights['Wb']
#                    break
            
            
            delta_Wb = np.dot(H.T, error)
            # print(error.shape)
            # print(delta_Wb.shape)
            
            delta_Wf = np.dot(X.T, np.dot(error, self.Wb.T))
            
            alpha_forward, alpha_backward = self.calculate_adaptive_learning_rate(X, H)
            
            self.Wb += alpha_backward * delta_Wb
            self.Wf += alpha_forward * delta_Wf
    
    def train_sample_by_sample(self, X, max_error, epochs=1000, target_mse=1e-5):
        best_error = float('inf')
        best_weights = {'Wf': None, 'Wb': None} 
        patience = 10
        no_improvement = 0
        
        total_pixels = X.shape[1] // 3  
        r = m = int(np.sqrt(total_pixels))  
        
        print(f"Размерность входного вектора X: {X.shape} (количество блоков × размер блока)")
        print(f"Где {X.shape[0]} = количество блоков ({256//8}x{256//8})")
        print(f"    {X.shape[1]} = размер блока ({r}x{m}x3 канала)")
        
        epoch = 0
        while best_error >= max_error and epoch < epochs:
            total_error = 0
            total_error_no_update = 0
            
            # Первый цикл - обучение
            for i in range(X.shape[0]):
                x_i = X[i:i+1]
                h_i = self.forward(x_i)
                x_reconstructed_i = self.backward(h_i)
                error_i = x_i - x_reconstructed_i
                
                delta_Wb = np.dot(h_i.T, error_i)
                delta_Wf = np.dot(x_i.T, np.dot(error_i, self.Wb.T))
                
                alpha_forward, alpha_backward = self.calculate_adaptive_learning_rate(x_i, h_i)
                
                self.Wb += alpha_backward * delta_Wb
                self.Wf += alpha_forward * delta_Wf

            # Второй цикл - подсчет ошибок
            for i in range(X.shape[0]):
                x_i = X[i:i+1]
                
                # Подсчет ошибки после обновления весов
                h_i = self.forward(x_i)
                x_reconstructed_i = self.backward(h_i)
                error_i = x_i - x_reconstructed_i
                total_error += self.calculate_mse(error_i)
                
                # Подсчет ошибки без обновления весов
                h_i_no_update = self.forward(x_i)
                x_reconstructed_i_no_update = self.backward(h_i_no_update)
                error_i_no_update = x_i - x_reconstructed_i_no_update
                total_error_no_update += self.calculate_mse(error_i_no_update)
            
            # Проверка MSE каждые 10 эпох
            if epoch % 10 == 0:
                current_mse = total_error 
                current_mse_no_update = total_error_no_update 
                self.mse_history.append(current_mse)
                print(f"Эпоха {epoch}:")
                print(f"  СК без обновления весов: {current_mse_no_update:.6f}")
                
                if current_mse < best_error:
                    best_error = current_mse
                    best_weights['Wf'] = self.Wf.copy()
                    best_weights['Wb'] = self.Wb.copy()
                    no_improvement = 0
                else:
                    no_improvement += 1
                
                if current_mse < target_mse or no_improvement >= patience:
                    print(f"Остановка на эпохе {epoch}")
                    self.Wf = best_weights['Wf']
                    self.Wb = best_weights['Wb']
                    break
            
            epoch += 1  # Увеличиваем номер эпохи

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path).resize(target_size)
    return np.array(img)

def split_image_to_rectangles(image, r, m):
    h, w, _ = image.shape
    rectangles = []
    for i in range(0, h, r):
        for j in range(0, w, m):
            if i + r <= h and j + m <= w:
                rectangle = image[i:i+r, j:j+m]
                rectangles.append(rectangle)
    return rectangles

def preprocess_color_values(C):
    return C.astype(np.float32) / 127.5 - 1.0

def create_feature_vector(rectangle):
    r, m, channels = rectangle.shape
    X = np.zeros((1, r * m * channels))
    idx = 0
    for i in range(r):
        for j in range(m):
            for c in range(channels):
                X[0, idx] = rectangle[i, j, c]
                idx += 1
    return X

def reconstruct_values(XX):
    XX_clipped = np.clip(XX, -1, 1)
    return ((XX_clipped + 1) * 127.5).astype(np.uint8)

def compress_image(image_path, r, m, p, max_error):
    img_array = load_and_preprocess_image(image_path)
    rectangles = split_image_to_rectangles(img_array, r, m)
    
    blocks = []
    expected_size = r * m * 3
    
    for rectangle in rectangles:
        normalized_rectangle = preprocess_color_values(rectangle)
        block_vector = create_feature_vector(normalized_rectangle)
        blocks.append(block_vector)
    
    X = np.vstack(blocks)
    
    network = RecyclingNetwork(X.shape[1], p)
    network.train_sample_by_sample(X, max_error)
    
    compressed_data = network.forward(X)
    
    h, w, _ = img_array.shape
    Q = (h // r) * (w // m)
    n = r * m * 3
    compression_ratio = (Q * 8) / (2 + (n + Q) * p)
    
    return network, compressed_data, compression_ratio

def reconstruct_image(compressed_data, h, w, r, m, network):
    channels = 3
    reconstructed_vectors = network.backward(compressed_data)
    block_size = r * m * channels  # Правильный размер блока
    
    reconstructed = []
    for block_vector in reconstructed_vectors:
        block = reconstruct_values(block_vector)
        block = block.reshape(r, m, channels)
        reconstructed.append(block)
    
    result = np.zeros((h, w, channels), dtype=np.uint8)
    block_idx = 0

    for i in range(0, h, r):
        for j in range(0, w, m):
            if block_idx < len(reconstructed) and i + r <= h and j + m <= w:
                result[i:i+r, j:j+m] = reconstructed[block_idx]
            block_idx += 1
    
    return result

def main():
    image_path = "cat.bmp"
    r, m = 8, 8
    p = 64
    max_error = 200
    
    network, compressed_data, compression_ratio = compress_image(image_path, r, m, p, max_error)
    print(f"Коэффициент сжатия: {compression_ratio:.2f}")
    
    reconstructed = reconstruct_image(compressed_data, 256, 256, r, m, network)
    reconstructed_image = Image.fromarray(reconstructed)
    reconstructed_image.save("обработанное.bmp")

if __name__ == "__main__":
    main()
