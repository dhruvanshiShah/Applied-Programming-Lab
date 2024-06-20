import os

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# List the files in the current directory
print("Files in Current Directory:")
for filename in os.listdir():
    print(filename)
