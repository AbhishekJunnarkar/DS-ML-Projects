# Matplotlib API Reference Guide for Data Exploration & Visualization

## 1. Importing Matplotlib
```python
import matplotlib.pyplot as plt
```

## 2. Basic Plot Types

- **Line Plot:**  
  `plt.plot(x, y)`

- **Scatter Plot:**  
  `plt.scatter(x, y)`

- **Bar Plot:**  
  `plt.bar(x, height)`  
  `plt.barh(y, width)`

- **Histogram:**  
  `plt.hist(data, bins=n)`

- **Box Plot:**  
  `plt.boxplot(data)`

- **Pie Chart:**  
  `plt.pie(sizes, labels=labels)`

## 3. Customizing Plots

- **Title:**  
  `plt.title('My Title')`

- **Axis Labels:**  
  `plt.xlabel('X Label')`  
  `plt.ylabel('Y Label')`

- **Legend:**  
  `plt.legend(['Series 1', 'Series 2'])`

- **Figure Size:**  
  `plt.figure(figsize=(width, height))`

- **Grid:**  
  `plt.grid(True)`

## 4. Displaying & Saving Plots

- **Show Plot:**  
  `plt.show()`

- **Save Plot:**  
  `plt.savefig('filename.png', dpi=300)`

## 5. Subplots

- **Multiple Subplots:**  
  `fig, axs = plt.subplots(nrows, ncols)`  
  `axs[0, 0].plot(x, y)`

- **Adjust Layout:**  
  `plt.tight_layout()`

## 6. Styling

- **Change Line Color, Style, and Marker:**  
  `plt.plot(x, y, color='r', linestyle='--', marker='o')`

- **Colormap (for scatter):**  
  `plt.scatter(x, y, c=values, cmap='viridis')`  
  `plt.colorbar()`

## 7. Common Plot Parameters

- **Alpha (transparency):**  
  `plt.plot(x, y, alpha=0.5)`

- **Limits:**  
  `plt.xlim(min, max)`  
  `plt.ylim(min, max)`

- **Ticks:**  
  `plt.xticks(ticks, labels)`  
  `plt.yticks(ticks, labels)`

## 8. Annotating Plots

- **Add Text:**  
  `plt.text(x, y, 'label')`

- **Annotate with Arrow:**  
  `plt.annotate('note', xy=(x, y), xytext=(x2, y2), arrowprops=dict(arrowstyle='->'))`

## 9. Working with Images

- **Display Image:**  
  `img = plt.imread('image.png')`  
  `plt.imshow(img)`

- **Remove Axis:**  
  `plt.axis('off')`

---

For more, visit the [Matplotlib Official Documentation](https://matplotlib.org/stable/contents.html)