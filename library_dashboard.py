import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class LibraryDashboard:

    def __init__(self):
        self.data = None

    # 1. Load and validate CSV data
    def load_data(self, file_path):

        if not os.path.exists(file_path):
            print("CSV file not found!")
            return False

        try:
            self.data = pd.read_csv(file_path)

            required_columns = [
                "Transaction_ID",
                "Date",
                "User_ID",
                "Book_Title",
                "Genre",
                "Borrowing_Duration_Days"
            ]

            for col in required_columns:
                if col not in self.data.columns:
                    print("Missing column:", col)
                    self.data = None
                    return False

            self.data["Date"] = pd.to_datetime(
                self.data["Date"], errors="coerce"
            )

            self.data["Borrowing_Duration_Days"] = pd.to_numeric(
                self.data["Borrowing_Duration_Days"],
                errors="coerce"
            )

            # Remove duplicate and invalid records
            self.data.drop_duplicates(inplace=True)

            self.data.dropna(
                subset=required_columns,
                inplace=True
            )

            self.data = self.data[
                self.data["Borrowing_Duration_Days"] > 0
            ].copy()

            self.data["Borrowing_Duration_Days"] = (
                self.data["Borrowing_Duration_Days"].astype(int)
            )

            print("Data loaded successfully!")
            print("Total records:", len(self.data))

            return True

        except (OSError, ValueError, pd.errors.ParserError) as error:
            print("Error loading file:", error)
            self.data = None
            return False

    # 2. Calculate statistics
    def calculate_statistics(self):

        if self.data is None or self.data.empty:
            print("No data available!")
            return

        duration = self.data[
            "Borrowing_Duration_Days"
        ].to_numpy()

        print("\n----- LIBRARY STATISTICS -----")

        print("Total Transactions:", len(self.data))

        print("Unique Users:",
              self.data["User_ID"].nunique())

        print("Unique Books:",
              self.data["Book_Title"].nunique())

        print("Average Borrowing Days:",
              round(np.mean(duration), 2))

        print("Median Borrowing Days:",
              np.median(duration))

        print("Standard Deviation:",
              round(np.std(duration), 2))

        print("\nMost Borrowed Books:")
        print(
            self.data["Book_Title"]
            .value_counts()
            .head(5)
        )

        print("\nBorrowings by Genre:")
        print(
            self.data.groupby("Genre")
            .size()
            .sort_values(ascending=False)
        )

        print("\nBusiest Day:",
              self.data["Date"].dt.day_name().mode()[0])

    # 3. Filter transactions
    def filter_transactions(self, genre=None,
                            start_date=None, end_date=None):

        if self.data is None:
            print("Load data first!")
            return

        result = self.data.copy()

        if genre and genre.lower() != "all":
            result = result[
                result["Genre"].str.lower() == genre.lower()
            ]

        if start_date:
            start = pd.to_datetime(start_date, errors="coerce")

            if pd.isna(start):
                print("Invalid start date!")
                return

            result = result[result["Date"] >= start]

        if end_date:
            end = pd.to_datetime(end_date, errors="coerce")

            if pd.isna(end):
                print("Invalid end date!")
                return

            result = result[result["Date"] <= end]

        print("\nFiltered Transactions:")
        print(result.head(20).to_string(index=False))
        print("Matching records:", len(result))

        return result

    # 4. Generate report
    def generate_report(self):

        if self.data is None or self.data.empty:
            print("No data available!")
            return

        with open("library_report.txt", "w",
                  encoding="utf-8") as file:

            file.write("E-LIBRARY DATA INSIGHTS REPORT\n")
            file.write("==============================\n\n")

            file.write(
                f"Total Transactions: {len(self.data)}\n"
            )

            file.write(
                f"Unique Users: {self.data['User_ID'].nunique()}\n"
            )

            file.write(
                "Average Borrowing Duration: "
                f"{self.data['Borrowing_Duration_Days'].mean():.2f} days\n\n"
            )

            file.write("TOP 5 BOOKS\n")

            file.write(
                self.data["Book_Title"]
                .value_counts()
                .head(5)
                .to_string()
            )

            file.write("\n\nGENRE SUMMARY\n")

            file.write(
                self.data.groupby("Genre")
                .agg(
                    Total_Borrowings=("Transaction_ID", "count"),
                    Average_Duration=("Borrowing_Duration_Days", "mean")
                )
                .round(2)
                .to_string()
            )

        print("Report saved as library_report.txt")

    # 5. Create visualizations
    def visualize_data(self):

        if self.data is None or self.data.empty:
            print("No data available!")
            return

        os.makedirs("library_charts", exist_ok=True)

        sns.set_theme(style="whitegrid")

        # Bar Chart: Top 5 Books
        top_books = self.data["Book_Title"].value_counts().head(5)

        plt.figure(figsize=(9, 5))
        sns.barplot(x=top_books.values, y=top_books.index)

        plt.title("Top 5 Most Borrowed Books")
        plt.xlabel("Number of Borrowings")
        plt.ylabel("Book Title")
        plt.tight_layout()

        plt.savefig("library_charts/top_books.png")
        plt.show()
        plt.close()

        # Line Graph: Monthly Borrowing Trends
        monthly = (
            self.data.set_index("Date")
            .resample("ME")
            .size()
        )

        plt.figure(figsize=(9, 5))
        plt.plot(monthly.index, monthly.values, marker="o")

        plt.title("Monthly Borrowing Trends")
        plt.xlabel("Month")
        plt.ylabel("Number of Borrowings")
        plt.tight_layout()

        plt.savefig("library_charts/monthly_trends.png")
        plt.show()
        plt.close()

        # Pie Chart: Genre Distribution
        genre_count = self.data["Genre"].value_counts()

        plt.figure(figsize=(7, 7))
        plt.pie(
            genre_count.values,
            labels=genre_count.index,
            autopct="%1.1f%%",
            startangle=90
        )

        plt.title("Books Borrowed by Genre")
        plt.tight_layout()

        plt.savefig("library_charts/genre_distribution.png")
        plt.show()
        plt.close()

        # Heatmap: Borrowing by Day and Month
        data = self.data.copy()

        data["Day"] = data["Date"].dt.day_name()
        data["Month"] = data["Date"].dt.strftime("%Y-%m")

        heatmap_data = data.pivot_table(
            index="Day",
            columns="Month",
            values="Transaction_ID",
            aggfunc="count",
            fill_value=0
        )

        days = [
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ]

        heatmap_data = heatmap_data.reindex(days).fillna(0)

        plt.figure(figsize=(11, 5))
        sns.heatmap(heatmap_data, annot=True, fmt=".0f",
                    cmap="YlGnBu")

        plt.title("Borrowing Activity by Day and Month")
        plt.xlabel("Month")
        plt.ylabel("Day")
        plt.tight_layout()

        plt.savefig("library_charts/borrowing_heatmap.png")
        plt.show()
        plt.close()

        print("All charts saved in library_charts folder!")


# Main program
def main():

    dashboard = LibraryDashboard()

    file_path = input(
        "Enter CSV file name (default: library_transactions.csv): "
    ).strip()

    if not file_path:
        file_path = "library_transactions.csv"

    if not dashboard.load_data(file_path):
        return

    while True:

        print("""
===== E-LIBRARY DATA INSIGHTS DASHBOARD =====
1. Show Statistics
2. Filter Transactions
3. Generate Report
4. Show Charts
5. Exit
""")

        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            dashboard.calculate_statistics()

        elif choice == "2":

            genre = input(
                "Enter genre (or all): "
            ).strip() or "all"

            start = input(
                "Enter start date YYYY-MM-DD (or Enter to skip): "
            ).strip() or None

            end = input(
                "Enter end date YYYY-MM-DD (or Enter to skip): "
            ).strip() or None

            dashboard.filter_transactions(
                genre=genre,
                start_date=start,
                end_date=end
            )

        elif choice == "3":
            dashboard.generate_report()

        elif choice == "4":
            dashboard.visualize_data()

        elif choice == "5":
            print("Thank you for using the dashboard!")
            break

        else:
            print("Invalid choice! Enter 1 to 5.")


if __name__ == "__main__":
    main()
