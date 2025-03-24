""" 
 Inputs:
   methods - Cell array of strings, each representing a control method.

 Outputs:
   ControlFlag - The index of the selected method.
"""
               
def control_method(methods):
    print("제어 방법을 선택하세요:")
    
    for i, method in enumerate(methods):
        print(f"{i+1}. {method}")
    
    while True:
        try:
            choice = int(input("번호를 입력하세요: "))
            if 1 <= choice <= len(methods):
                return choice   # 1부터 시작하는 인덱스를 반환
            else:
                print("유효한 번호를 입력하세요.")
        except ValueError:
            print("정수를 입력하세요.")

