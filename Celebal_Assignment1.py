# Function to print lower triangular pattern
def lower_triangular(n):
    print("Lower Triangular Pattern:")
    for i in range(n):
        print("*" * (i + 1))

# Function to print upper triangular pattern
def upper_triangular(n):
    print("\nUpper Triangular Pattern:")
    for i in range(n):
        print(" " * i + "*" * (n - i))

# Function to print pyramid pattern
def pyramid(n):
    print("\nPyramid Pattern:")
    for i in range(n):
        print(" " * (n - i - 1) + "*" * (2 * i + 1))

# Get number of rows from user
try:
    rows = int(input("Enter the number of rows: "))
    if rows <= 0:
        print("Please enter a positive number of rows.")
    else:
        # Call functions
        lower_triangular(rows)
        upper_triangular(rows)
        pyramid(rows)
except ValueError:
    print("Please enter a valid integer.")
