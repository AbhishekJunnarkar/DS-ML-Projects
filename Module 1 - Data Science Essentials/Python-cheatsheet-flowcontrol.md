**Python Cheatsheet for Flow Control** that you can use for quick daily revision:

---

### **Flow Control in Python**

---

#### **1. Conditional Statements**

##### **if Statement**
Executes a block of code if the condition is `True`.

```python
if condition:
    # Code to execute
```

Example:
```python
x = 10
if x > 5:
    print("x is greater than 5")
```

---

##### **if-else Statement**
Provides an alternative block of code if the condition is `False`.

```python
if condition:
    # Code if condition is True
else:
    # Code if condition is False
```

Example:
```python
x = 10
if x > 15:
    print("x is greater than 15")
else:
    print("x is not greater than 15")
```

---

##### **if-elif-else Statement**
Used to check multiple conditions.

```python
if condition1:
    # Code if condition1 is True
elif condition2:
    # Code if condition2 is True
else:
    # Code if all conditions are False
```

Example:
```python
x = 10
if x > 15:
    print("x is greater than 15")
elif x > 5:
    print("x is greater than 5 but not greater than 15")
else:
    print("x is less than or equal to 5")
```

---

##### **Nested if**
Conditions can be nested for more complex decision-making.

```python
if condition1:
    if condition2:
        # Code if both conditions are True
```

Example:
```python
x = 10
if x > 5:
    if x % 2 == 0:
        print("x is greater than 5 and even")
```

---

#### **2. Loops**

##### **for Loop**
Iterates over a sequence (e.g., list, tuple, string).

```python
for variable in sequence:
    # Code to execute
```

Example:
```python
fruits = ['apple', 'banana', 'cherry']
for fruit in fruits:
    print(fruit)
```

---

##### **range() in for Loop**
Generates a sequence of numbers.

```python
for i in range(start, stop, step):
    # Code to execute
```

Examples:
```python
for i in range(5):  # 0, 1, 2, 3, 4
    print(i)

for i in range(1, 10, 2):  # 1, 3, 5, 7, 9
    print(i)
```

---

##### **while Loop**
Executes as long as the condition is `True`.

```python
while condition:
    # Code to execute
```

Example:
```python
x = 0
while x < 5:
    print(x)
    x += 1
```

---

#### **3. Loop Control Statements**

##### **break**
Exits the loop prematurely.

```python
for i in range(10):
    if i == 5:
        break
    print(i)
```

Output:
```
0 1 2 3 4
```

---

##### **continue**
Skips the current iteration and proceeds to the next.

```python
for i in range(10):
    if i % 2 == 0:
        continue
    print(i)
```

Output:
```
1 3 5 7 9
```

---

##### **pass**
Does nothing; used as a placeholder.

```python
for i in range(5):
    if i == 3:
        pass  # Placeholder
    print(i)
```

---

##### **else with Loops**
The `else` block runs if the loop completes normally (without `break`).

```python
for i in range(5):
    print(i)
else:
    print("Loop completed without break")
```

Example with `break`:
```python
for i in range(5):
    if i == 3:
        break
else:
    print("This will not print because loop was broken")
```

---

#### **4. Pattern Printing with Loops**

Example:
```python
rows = 5
for i in range(1, rows + 1):
    print("*" * i)
```

Output:
```
*
**
***
****
*****
```

---

#### **5. Exception Handling in Flow Control**

##### **try-except**
Handles exceptions during program execution.

```python
try:
    # Code that might raise an exception
except ExceptionType:
    # Code to execute if an exception occurs
```

Example:
```python
try:
    x = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
```

---

##### **try-except-else**
Executes the `else` block if no exception occurs.

```python
try:
    # Code that might raise an exception
except ExceptionType:
    # Code to execute if an exception occurs
else:
    # Code to execute if no exception occurs
```

---

##### **try-except-finally**
Executes the `finally` block regardless of exceptions.

```python
try:
    # Code that might raise an exception
except ExceptionType:
    # Code to execute if an exception occurs
finally:
    # Code to execute no matter what
```

Example:
```python
try:
    x = int("10")
except ValueError:
    print("Invalid input")
finally:
    print("This will always execute")
```

---
