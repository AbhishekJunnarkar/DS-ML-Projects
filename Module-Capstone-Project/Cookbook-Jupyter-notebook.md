# Jupyter Notebook Cookbook for AI/ML Projects

This guide offers practical recipes for accessing, using, and optimizing Jupyter Notebooks in AI/ML workflows.

---

## 1. Getting Started

### 1.1 Installing Jupyter Notebook

```bash
pip install notebook
# or, for JupyterLab:
pip install jupyterlab
```

### 1.2 Launching Jupyter Notebook

```bash
jupyter notebook
# Or for JupyterLab:
jupyter lab
```
- Opens in your default browser at `http://localhost:8888`.

---

## 2. Basic Usage

### 2.1 Creating a New Notebook

- In the Jupyter dashboard, click "New" → "Python 3" (or desired kernel).

### 2.2 Notebook Cells

- **Code Cell:** Run Python code.
- **Markdown Cell:** Write formatted text, equations, or documentation.

**Run a cell:**  
- Click cell and press `Shift + Enter` or click the "Run" button.

---

## 3. Essential Shortcuts

- `Shift + Enter`: Run cell & move to next
- `Ctrl + Enter`: Run cell, stay in place
- `A` / `B`: Insert cell Above / Below (in command mode)
- `M`: Convert to Markdown
- `Y`: Convert to Code
- `D, D`: Delete cell (press D twice)
- `Z`: Undo cell deletion
- `Esc`: Enter command mode
- `Enter`: Enter edit mode

---

## 4. Key Features for AI/ML Projects

### 4.1 Importing Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
```

### 4.2 Data Upload & Inspection

```python
df = pd.read_csv('data.csv')
df.head()
df.info()
```

### 4.3 Visualizations Inline

```python
%matplotlib inline
plt.figure(figsize=(8,5))
sns.histplot(df['feature'])
plt.show()
```

### 4.4 Using Magic Commands

- `%matplotlib inline` — Plot graphs inside notebook
- `%timeit my_function()` — Time code execution
- `%ls` — List directory contents
- `%pwd` — Print working directory

### 4.5 Exporting Your Work

- **Download as .ipynb or .py:**  
  File → Download as → Notebook (.ipynb) or Python (.py)

---

## 5. Version Control & Collaboration

- **Use JupyterLab or GitHub for collaboration.**
- Save checkpoints frequently:  
  File → Save and Checkpoint
- Version control notebooks by exporting as `.py` scripts or using [nbdime](https://nbdime.readthedocs.io/) for notebook diffs.

---

## 6. Best Practices

- **Document your workflow:** Use Markdown cells for explanations.
- **Seed randomness:**  
  `np.random.seed(42)` for reproducibility.
- **Clear outputs before sharing:**  
  Kernel → Restart & Clear Output.
- **Break code into small cells:** Easier to debug and rerun.

---

## 7. Troubleshooting

- **Kernel busy or unresponsive:**  
  Kernel → Restart.
- **Package not found:**  
  Use `!pip install package_name` in a cell.
- **Plots not showing:**  
  Add `%matplotlib inline` at the top.

---

## 8. Resources

- [Jupyter Documentation](https://jupyter.org/documentation)
- [nbextensions](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/): Power up your notebook!
- [Google Colab](https://colab.research.google.com/): Free cloud Jupyter notebooks

---

Happy Coding!