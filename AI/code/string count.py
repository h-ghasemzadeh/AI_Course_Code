import random
import string
from collections import Counter

# Generate a random string of length 80
random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=80))

# Print the random string
print("Random String:", random_string)

# Count the repetition of each character
char_count = Counter(random_string)

# Print the repetition of each character
print("\nCharacter Repetition:")
for char, count in char_count.items():
    print(f"'{char}': {count} times")