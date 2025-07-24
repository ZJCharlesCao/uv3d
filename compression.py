import os
from pathlib import Path
from PIL import Image  
import numpy as np
import cv2 
import io

class ImageCompressor:
    def __init__(self): 
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    def load_image(self, image_path: str) -> np.ndarray:
        """加载图像"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 使用OpenCV加载图像
        image = cv2.imread(image_path)
        if image is None:
            # 尝试使用PIL加载
            pil_image = Image.open(image_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return image
    
    def compress_jpeg(self, image: np.ndarray, quality: int = 100) -> bytes:
        """
        JPEG压缩
        quality: 1-100, 数值越高质量越好
        """
        # 转换为PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        
        buffer = io.BytesIO()
        # 保存为JPEG格式
        pil_image.save(buffer, format='JPEG', quality=quality)
        
        return buffer.getvalue()
    
    def compress_png(self, image: np.ndarray, compress_level: int = 6) -> bytes:
        """
        PNG压缩 - 无损压缩
        compress_level: 0-9, 数值越高压缩率越高但速度越慢
        """
        # 使用OpenCV压缩
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, compress_level]
        success, encoded_img = cv2.imencode('.png', image, encode_params)
        
        if success:
            return encoded_img.tobytes()
        else:
            raise RuntimeError("PNG压缩失败")
        
    def batch_compress(self, input_folder: str, output_folder: str):
        """批量压缩文件夹中的图像，分别创建jpeg和png子目录"""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        # 创建输出目录和子目录
        output_path.mkdir(exist_ok=True)
        jpeg_dir = output_path / "jpeg"
        png_dir = output_path / "png"
        jpeg_dir.mkdir(exist_ok=True)
        png_dir.mkdir(exist_ok=True)
        
        # 获取所有支持的图像文件
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print("在输入目录中未找到支持的图像文件")
            return
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        for image_file in image_files:
            try:
                # 加载图像
                image = self.load_image(str(image_file))
                original_size = os.path.getsize(image_file)
                
                # JPEG压缩
                try:
                    jpeg_data = self.compress_jpeg(image, quality=100)
                    jpeg_file = jpeg_dir / f"{image_file.stem}.jpeg"
                    with open(jpeg_file, 'wb') as f:
                        f.write(jpeg_data)
                    
                    jpeg_size = len(jpeg_data)
                    jpeg_saved = (1 - jpeg_size / original_size) * 100
                    print(f"JPEG: {image_file.name} -> {jpeg_file.name} "
                          f"({original_size/1024:.1f}KB -> {jpeg_size/1024:.1f}KB, "
                          f"节省{jpeg_saved:.1f}%)")
                    
                except Exception as e:
                    print(f"JPEG压缩失败 {image_file.name}: {e}")
                
                # PNG压缩
                try:
                    png_data = self.compress_png(image, compress_level=6)
                    png_file = png_dir / f"{image_file.stem}.png"
                    with open(png_file, 'wb') as f:
                        f.write(png_data)
                    
                    png_size = len(png_data)
                    png_saved = (1 - png_size / original_size) * 100
                    print(f"PNG: {image_file.name} -> {png_file.name} "
                          f"({original_size/1024:.1f}KB -> {png_size/1024:.1f}KB, "
                          f"节省{png_saved:.1f}%)")
                    
                except Exception as e:
                    print(f"PNG压缩失败 {image_file.name}: {e}")
                
                print()
                
            except Exception as e:
                print(f"处理失败 {image_file.name}: {e}")
    
    def compress_single_image(self, image_path: str, output_dir: str = None):
        """压缩单个图像文件"""
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 加载图像
        image = self.load_image(image_path)
        original_size = os.path.getsize(image_path)
        base_name = Path(image_path).stem
        
        print(f"原始图像: {os.path.basename(image_path)} ({original_size/1024:.1f}KB)")
        
        # JPEG压缩
        try:
            jpeg_data = self.compress_jpeg(image, quality=85)
            jpeg_file = output_path / f"{base_name}_compressed.jpeg"
            with open(jpeg_file, 'wb') as f:
                f.write(jpeg_data)
            
            jpeg_size = len(jpeg_data)
            jpeg_saved = (1 - jpeg_size / original_size) * 100
            print(f"JPEG: {jpeg_file.name} ({jpeg_size/1024:.1f}KB, 节省{jpeg_saved:.1f}%)")
            
        except Exception as e:
            print(f"JPEG压缩失败: {e}")
        
        # PNG压缩
        try:
            png_data = self.compress_png(image, compress_level=6)
            png_file = output_path / f"{base_name}_compressed.png"
            with open(png_file, 'wb') as f:
                f.write(png_data)
            
            png_size = len(png_data)
            png_saved = (1 - png_size / original_size) * 100
            print(f"PNG: {png_file.name} ({png_size/1024:.1f}KB, 节省{png_saved:.1f}%)")
            
        except Exception as e:
            print(f"PNG压缩失败: {e}")


