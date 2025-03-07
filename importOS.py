import os

annotation_dir = 'C:/Users/Innomaker Dev/Downloads/coral-sol/test/labels'

target_class = 0

for filename in os.listdir(annotation_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(annotation_dir, filename)
              
        with open(filepath, 'r') as file:
            lines = file.readlines()
        
        with open(filepath, 'w') as file:
            for line in lines:
                parts = line.strip().split()
                
                parts[0] = str(target_class)
                
                file.write(" ".join(parts) + '\n')

print("Todas as anotações foram alteradas para a classe 'coral'.")
