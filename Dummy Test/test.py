import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
import data_preprocessing_functions as dpf

class DataPreprocessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Preprocessor")
        self.root.geometry("1200x1100")
        self.root.resizable(False, False)

        self.data = None

        # Main layout configuration
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=3)
        self.root.columnconfigure(2, weight=2)
        self.root.rowconfigure(0, weight=1)

        # Left frame: Buttons
        self.left_frame = ttk.Frame(root, padding="10")
        self.left_frame.grid(row=0, column=0, sticky="nsw")

        self.file_label = ttk.Label(self.left_frame, text="File Operations")
        self.file_label.grid(row=0, column=0, pady=(0, 5))

        self.file_button = ttk.Button(self.left_frame, text="Load CSV File", command=self.load_file)
        self.file_button.grid(row=1, column=0, sticky="ew", pady=5)

        self.info_button = ttk.Button(self.left_frame, text="Show Data Info", command=self.show_info, state=tk.DISABLED)
        self.info_button.grid(row=2, column=0, sticky="ew", pady=5)

        self.export_button = ttk.Button(self.left_frame, text="Export to CSV", command=self.export_to_csv, state=tk.DISABLED)
        self.export_button.grid(row=3, column=0, sticky="ew", pady=5)

        ttk.Separator(self.left_frame).grid(row=4, column=0, sticky="ew", pady=10)

        self.column_label = ttk.Label(self.left_frame, text="Column Operations")
        self.column_label.grid(row=5, column=0, pady=(0, 5))

        self.remove_button = ttk.Button(self.left_frame, text="Remove Rows with Null Values", command=self.remove_null_rows, state=tk.DISABLED)
        self.remove_button.grid(row=6, column=0, sticky="ew", pady=5)

        self.fill_button = ttk.Button(self.left_frame, text="Fill Nulls with Mean", command=self.fill_nulls_with_mean, state=tk.DISABLED)
        self.fill_button.grid(row=7, column=0, sticky="ew", pady=5)

        self.one_hot_button = ttk.Button(self.left_frame, text="One Hot Encode Column", command=self.one_hot_encode, state=tk.DISABLED)
        self.one_hot_button.grid(row=8, column=0, sticky="ew", pady=5)

        self.ordinal_button = ttk.Button(self.left_frame, text="Ordinal Encode Column", command=self.ordinal_encode, state=tk.DISABLED)
        self.ordinal_button.grid(row=9, column=0, sticky="ew", pady=5)

        self.unique_button = ttk.Button(self.left_frame, text="View Unique Values", command=self.view_unique_values, state=tk.DISABLED)
        self.unique_button.grid(row=10, column=0, sticky="ew", pady=5)

        self.boolean_button = ttk.Button(self.left_frame, text="Convert Boolean to 0/1", command=self.convert_boolean, state=tk.DISABLED)
        self.boolean_button.grid(row=11, column=0, sticky="ew", pady=5)

        self.date_button = ttk.Button(self.left_frame, text="Convert to Date", command=self.convert_to_date, state=tk.DISABLED)
        self.date_button.grid(row=12, column=0, sticky="ew", pady=5)

        # Center frame: Column status
        self.center_frame = ttk.Frame(root, padding="10")
        self.center_frame.grid(row=0, column=1, sticky="nsew")

        self.column_status_label = ttk.Label(self.center_frame, text="Column Status:")
        self.column_status_label.grid(row=0, column=0, sticky="w", pady=5)

        self.column_tree = ttk.Treeview(self.center_frame, columns=("name", "type", "nulls"), show="headings", height=30)
        self.column_tree.heading("name", text="Column Name")
        self.column_tree.heading("type", text="Data Type")
        self.column_tree.heading("nulls", text="Null Count")
        self.column_tree.column("name", width=200)
        self.column_tree.column("type", width=100)
        self.column_tree.column("nulls", width=100)
        self.column_tree.grid(row=1, column=0, sticky="nsew", pady=5)

        self.center_frame.rowconfigure(1, weight=0)
        self.center_frame.columnconfigure(0, weight=1)

        # Right frame: Log
        self.right_frame = ttk.Frame(root, padding="10")
        self.right_frame.grid(row=0, column=2, sticky="nsew")

        self.log_label = ttk.Label(self.right_frame, text="Action Log:")
        self.log_label.grid(row=0, column=0, sticky="w", pady=5)

        self.log_text = tk.Text(self.right_frame, wrap=tk.WORD, state=tk.DISABLED, height=30, width=40)
        self.log_text.grid(row=1, column=0, sticky="nsew", pady=5)

        self.right_frame.rowconfigure(1, weight=0)
        self.right_frame.columnconfigure(0, weight=1)

    def log_message(self, message, error=False):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, ("ERROR: " if error else "") + message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.data = dpf.read_csv(file_path, sep=",")
                self.update_column_status()
                self.log_message("File loaded successfully!")
                self.enable_buttons()
            except Exception as e:
                self.log_message(f"Failed to load file: {e}", error=True)

    def enable_buttons(self):
        self.info_button.config(state=tk.NORMAL)
        self.remove_button.config(state=tk.NORMAL)
        self.fill_button.config(state=tk.NORMAL)
        self.one_hot_button.config(state=tk.NORMAL)
        self.ordinal_button.config(state=tk.NORMAL)
        self.unique_button.config(state=tk.NORMAL)
        self.boolean_button.config(state=tk.NORMAL)
        self.date_button.config(state=tk.NORMAL)
        self.export_button.config(state=tk.NORMAL)

    def update_column_status(self):
        if self.data is not None:
            for row in self.column_tree.get_children():
                self.column_tree.delete(row)
            for column in self.data.columns:
                null_count = self.data[column].isnull().sum()
                dtype = self.data[column].dtype
                self.column_tree.insert("", tk.END, values=(column, dtype, null_count))

    def get_selected_column(self):
        selected_item = self.column_tree.selection()
        if selected_item:
            return self.column_tree.item(selected_item[0], "values")[0]
        else:
            self.log_message("No column selected.", error=True)
            return None

    def show_info(self):
        if self.data is not None:
            info_window = tk.Toplevel(self.root)
            info_window.title("Data Info")
            info_window.geometry("600x400")

            text = tk.Text(info_window, wrap=tk.WORD, padx=10, pady=10)
            text.insert(tk.END, "First 5 rows:\n" + self.data.head().to_string() + "\n\nLast 5 rows:\n" + self.data.tail().to_string())
            text.config(state=tk.DISABLED)
            text.pack(expand=True, fill=tk.BOTH)

    def remove_null_rows(self):
        column = self.get_selected_column()
        if self.data is not None and column:
            self.data = dpf.remove_row_with_null_values(self.data, column)
            self.update_column_status()
            self.log_message(f"Rows with null values in '{column}' removed.")

    def fill_nulls_with_mean(self):
        column = self.get_selected_column()
        if self.data is not None and column:
            self.data = dpf.set_null_values_as_mean(self.data, column)
            self.update_column_status()
            self.log_message(f"Null values in '{column}' filled with column mean.")

    def one_hot_encode(self):
        column = self.get_selected_column()
        if self.data is not None and column:
            try:
                self.data = pd.get_dummies(self.data, columns=[column], dtype=int)
                self.update_column_status()
                self.log_message(f"Column '{column}' one-hot encoded successfully.")
            except Exception as e:
                self.log_message(f"Failed to one-hot encode column '{column}': {e}", error=True)

    def ordinal_encode(self):
        column = self.get_selected_column()
        if self.data is not None and column:
            try:
                unique_values = {val: i for i, val in enumerate(sorted(self.data[column].dropna().unique()))}
                self.data[column] = self.data[column].map(unique_values)
                self.update_column_status()
                self.log_message(f"Column '{column}' ordinal encoded successfully.")
            except Exception as e:
                self.log_message(f"Failed to ordinal encode column '{column}': {e}", error=True)

    def view_unique_values(self):
        column = self.get_selected_column()
        if self.data is not None and column:
            try:
                unique_values = self.data[column].dropna().unique()
                unique_window = tk.Toplevel(self.root)
                unique_window.title(f"Unique Values in '{column}'")
                unique_window.geometry("400x300")

                text = tk.Text(unique_window, wrap=tk.WORD, padx=10, pady=10)
                text.insert(tk.END, "\n".join(map(str, unique_values)))
                text.config(state=tk.DISABLED)
                text.pack(expand=True, fill=tk.BOTH)
            except Exception as e:
                self.log_message(f"Failed to retrieve unique values for column '{column}': {e}", error=True)

    def convert_boolean(self):
        column = self.get_selected_column()
        if self.data is not None and column:
            try:
                if self.data[column].dtype == bool or self.data[column].dropna().isin([True, False]).all():
                    self.data[column] = self.data[column].astype(int)
                    self.update_column_status()
                    self.log_message(f"Boolean column '{column}' converted to 0/1 successfully.")
                else:
                    self.log_message(f"Column '{column}' is not boolean.", error=True)
            except Exception as e:
                self.log_message(f"Failed to convert boolean column '{column}': {e}", error=True)

    def convert_to_date(self):
        column = self.get_selected_column()
        if self.data is not None and column:
            format_window = tk.Toplevel(self.root)
            format_window.title("Specify Date Format")
            format_window.geometry("400x150")

            ttk.Label(format_window, text="Enter date format (e.g., %Y-%m-%d):").pack(pady=10)
            format_entry = ttk.Entry(format_window, width=30)
            format_entry.pack(pady=5)

            def apply_conversion():
                date_format = format_entry.get()
                try:
                    self.data[column] = pd.to_datetime(self.data[column], format=date_format)
                    self.update_column_status()
                    self.log_message(f"Column '{column}' converted to datetime.")
                    format_window.destroy()
                except Exception as e:
                    self.log_message(f"Failed to convert column '{column}' to datetime: {e}", error=True)

            ttk.Button(format_window, text="Convert", command=apply_conversion).pack(pady=10)

    def export_to_csv(self):
        if self.data is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_path:
                try:
                    self.data.to_csv(file_path, index=False)
                    self.log_message("Data exported successfully!")
                except Exception as e:
                    self.log_message(f"Failed to export data: {e}", error=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = DataPreprocessorApp(root)
    root.mainloop()
