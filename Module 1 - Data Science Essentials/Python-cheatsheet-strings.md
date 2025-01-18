**Cheatsheet of all string functions in Python** for quick daily revision:

---

### **String Creation and Basic Operations**
- **`len(string)`**: Returns the length of the string.
- **`string[index]`**: Access a specific character by index.
- **`string[start:end:step]`**: Slicing a string with optional step.

---

### **String Methods**

#### **1. Formatting**
- **`string.lower()`**: Converts all characters to lowercase.
- **`string.upper()`**: Converts all characters to uppercase.
- **`string.title()`**: Converts the first character of each word to uppercase.
- **`string.capitalize()`**: Capitalizes the first character of the string.
- **`string.swapcase()`**: Swaps uppercase to lowercase and vice versa.
- **`string.center(width, fillchar)`**: Centers the string with a specified width and optional fill character.
- **`string.ljust(width, fillchar)`**: Left-aligns the string with a specified width.
- **`string.rjust(width, fillchar)`**: Right-aligns the string with a specified width.
- **`string.zfill(width)`**: Pads the string with zeros to the left.

---

#### **2. Stripping and Padding**
- **`string.strip(chars)`**: Removes leading and trailing characters (default: spaces).
- **`string.lstrip(chars)`**: Removes leading characters.
- **`string.rstrip(chars)`**: Removes trailing characters.

---

#### **3. Searching and Checking**
- **`string.find(substring, start, end)`**: Returns the index of the first occurrence of a substring; returns `-1` if not found.
- **`string.rfind(substring, start, end)`**: Returns the index of the last occurrence of a substring; returns `-1` if not found.
- **`string.index(substring, start, end)`**: Like `find()` but raises a `ValueError: substring not found` if not found.
- **`string.rindex(substring, start, end)`**: Like `rfind()` but raises a `ValueError: substring not found` if not found.
- **`string.startswith(substring, start, end)`**: Returns `True` if the string starts with the specified substring.
- **`string.endswith(substring, start, end)`**: Returns `True` if the string ends with the specified substring.

---

#### **4. Replacing and Joining**
- **`string.replace(old, new, count)`**: Replaces occurrences of a substring with another substring.
- **`string.join(iterable)`**: Joins elements of an iterable (e.g., list) into a single string, separated by the string.

---

#### **5. Splitting and Partitioning**
- **`string.split(separator, maxsplit)`**: Splits the string into a list using a separator.
- **`string.rsplit(separator, maxsplit)`**: Splits the string from the right.
- **`string.splitlines(keepends)`**: Splits the string at line breaks (`\n`).
- **`string.partition(separator)`**: Splits the string into a 3-tuple: `(before, separator, after)`.
- **`string.rpartition(separator)`**: Like `partition()` but splits from the right.

---

#### **6. Character Checks**
- **`string.isalpha()`**: Returns `True` if all characters are alphabetic.
- **`string.isdigit()`**: Returns `True` if all characters are digits.
- **`string.isalnum()`**: Returns `True` if all characters are alphanumeric.
- **`string.isspace()`**: Returns `True` if all characters are whitespace.
- **`string.islower()`**: Returns `True` if all characters are lowercase.
- **`string.isupper()`**: Returns `True` if all characters are uppercase.
- **`string.istitle()`**: Returns `True` if the string is in title case.

---

#### **7. Encoding and Decoding**
- **`string.encode(encoding, errors)`**: Encodes the string to bytes using the specified encoding.
- **`bytes.decode(encoding, errors)`**: Decodes bytes to a string using the specified encoding.

---

#### **8. Other Methods**
- **`string.count(substring, start, end)`**: Counts occurrences of a substring. e.g. string ABBCDDEEBB
  - s.count('B')
  - s.count('B',4,10)
- **`string.expandtabs(tabsize)`**: Replaces tabs with spaces.
- **`string.casefold()`**: Converts the string to lowercase (stronger than `lower()` for certain languages).
- **`string.format(*args, **kwargs)`**: Formats the string using placeholders.
- **`string.format_map(mapping)`**: Formats the string using a mapping (e.g., dictionary).
- **`string.maketrans(x, y, z)`**: Creates a translation table for `translate()`.
- **`string.translate(table)`**: Translates the string using a translation table.

---

### **String Constants**
- **`string.ascii_letters`**: All ASCII letters (lowercase + uppercase).
- **`string.ascii_lowercase`**: All ASCII lowercase letters.
- **`string.ascii_uppercase`**: All ASCII uppercase letters.
- **`string.digits`**: All numeric digits (`0-9`).
- **`string.punctuation`**: All punctuation characters.
- **`string.whitespace`**: All whitespace characters.

---
