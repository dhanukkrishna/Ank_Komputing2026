print("Checking the git repository for any issues...")

# Adding for lines
# Just for check


def main():
    try:
        a = int(input("Enter a number: "))
        b = int(input("Enter a number: "))
    except (EOFError, ValueError):
        print("Please enter valid integers only.")
        return

    print("The sum of the two numbers is:", a + b)


if __name__ == "__main__":
    main()


