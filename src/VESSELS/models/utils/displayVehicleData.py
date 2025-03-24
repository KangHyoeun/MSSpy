import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def display_vehicle_data(vehicleName, vehicleData, imageFile, figNo):
    """
     displays the vehicle main characteristics and an image of the vehicle.
    
     Inputs:
       vehicleName - A string representing the name of the vehicle 
                     (e.g., 'Remus 100 AUV')
       vehicleData - A cell array of key-value pairs representing the vehicle's 
                     characteristics (e.g., {'Length', '1.6 m', 
                     'Diameter', '19 cm', 'Mass', '31.9 kg', ...})
       imageFile   - A string representing the file name of the vehicle image 
                     located in .../MSS/SIMULINK/figs/ (e.g., 'remus100.jpg')
       figNo       - A number representing the figure number to use for the display
    
     Outputs:
       None
    """
    
    # 텍스트 생성
    heading = f"MSS Toolbox\n{vehicleName}"
    
    # 데이터 텍스트 생성
    data_text = "\n".join(f"{key} : {value}" for key, value in vehicleData)
    
    # 이미지 파일 경로 설정
    # 경로를 적절히 수정하세요
    image_path = f"MSS/src/SIMULINK/figs/{imageFile}"
    
    # figure 생성
    plt.figure(figNo)
    
    # 제목 표시
    plt.subplot(3, 1, 1)
    plt.text(0.1, 0.8, heading, fontsize=20)
    plt.axis('off')
    
    # 데이터 텍스트 표시
    plt.subplot(3, 1, 2)
    plt.text(0.1, 0.9, data_text, fontsize=16)
    plt.axis('off')
    
    # 이미지 표시
    plt.subplot(3, 1, 3)
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
