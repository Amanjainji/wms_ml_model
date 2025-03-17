import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Wedge

# Set the style for all plots
plt.style.use('default')
sns.set_style("whitegrid")

# Set random seed for reproducibility
np.random.seed(42)

def plot_all_charts(excel_file='waste_management_data.xlsx'):
    # Read data from Excel file
    print(f"Reading data from {excel_file}...")
    
    # Read each sheet from the Excel file
    main_data = pd.read_excel(excel_file, sheet_name='Waste Management Data')
    composition_data = pd.read_excel(excel_file, sheet_name='Waste Composition')
    efficiency_data = pd.read_excel(excel_file, sheet_name='Area Efficiency')
    bin_capacity_data = pd.read_excel(excel_file, sheet_name='Bin Capacity')
    
    print(f"Successfully loaded data with {len(main_data)} entries")
    
    # Process data for Waste Collection Trends Chart
    # Aggregate the data by calculating the average for each waste type
    # We'll create 9 categories (similar to the original 9 months)
    # by splitting the data into 9 equal groups
    
    # Create a new column with category numbers 1-9
    main_data['Category'] = pd.qcut(main_data['ID'], 9, labels=False) + 1
    
    # Group by category and calculate mean for each waste type
    trends_data = main_data.groupby('Category').agg({
        'GeneralWaste': 'mean',
        'Recyclable': 'mean',
        'OrganicWaste': 'mean',
        'HazardousWaste': 'mean'
    }).reset_index()
    
    # Convert category to string for plotting
    trends_data['Category'] = trends_data['Category'].astype(str)
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 16), facecolor='white')
    
    # 1. Waste Collection Trends Chart
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(trends_data['Category'], trends_data['GeneralWaste'], 'o-', color='#FF3366', linewidth=2, label='General Waste')
    ax1.plot(trends_data['Category'], trends_data['Recyclable'], 'o-', color='#33CC66', linewidth=2, label='Recyclable')
    ax1.plot(trends_data['Category'], trends_data['OrganicWaste'], 'o-', color='#FFCC33', linewidth=2, label='Organic')
    ax1.plot(trends_data['Category'], trends_data['HazardousWaste'], 'o-', color='#9966FF', linewidth=2, label='Hazardous')
    
    ax1.set_title('Waste Collection Trends Chart', fontsize=16, pad=20)
    ax1.set_xlabel('Category', fontsize=12)
    ax1.set_ylabel('Volume (tons)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_facecolor('#0A1929')
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=12)
    ax1.set_ylim(0, 600)
    
    # 2. Waste Composition Chart (Pie Chart)
    ax2 = plt.subplot(2, 2, 2)
    colors = ['#FF3366', '#33CC66', '#FFCC33', '#9966FF']
    
    # Extract data for pie chart
    waste_types = composition_data['WasteType'].tolist()
    percentages = composition_data['Percentage'].tolist()
    
    wedges, texts, autotexts = ax2.pie(
        percentages, 
        labels=None,
        colors=colors, 
        autopct='',
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )
    
    # Add percentage labels outside the pie
    for i, p in enumerate(percentages):
        ang = (wedges[i].theta2 - wedges[i].theta1) / 2. + wedges[i].theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        ax2.annotate(
            f"{waste_types[i]} {p}%",
            xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
            horizontalalignment=horizontalalignment,
            color=colors[i],
            fontsize=12
        )
    
    ax2.set_title('Waste Composition Chart', fontsize=16, pad=20)
    ax2.set_facecolor('#0A1929')
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=colors[i], markersize=10, 
                                 label=waste_types[i])
                      for i in range(len(waste_types))]
    ax2.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.6, 0.5), fontsize=12)
    
    # 3. Collection Efficiency by Area Chart (Bar Chart)
    ax3 = plt.subplot(2, 2, 3)
    bar_width = 0.35
    
    # Extract data for bar chart
    areas = efficiency_data['Area'].tolist()
    current_efficiency = efficiency_data['CurrentEfficiency'].tolist()
    target_efficiency = efficiency_data['TargetEfficiency'].tolist()
    
    x = np.arange(len(areas))
    
    ax3.bar(x - bar_width/2, current_efficiency, bar_width, color='#9966FF', label='Current Efficiency')
    ax3.bar(x + bar_width/2, target_efficiency, bar_width, color='#33CCFF', label='Target')
    
    ax3.set_title('Collection Efficiency by Area Chart', fontsize=16, pad=20)
    ax3.set_xlabel('Area', fontsize=12)
    ax3.set_ylabel('Efficiency (%)', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(areas)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_facecolor('#0A1929')
    ax3.set_ylim(80, 100)
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12)
    
    # 4. Bin Capacity Utilization Chart (Gauge Chart)
    ax4 = plt.subplot(2, 2, 4)
    
    def gauge(pos, value, color, title):
        # Gauge chart parameters
        r_outer = 0.8
        r_inner = 0.6
        
        # Create the gauge
        start_angle = 180
        end_angle = start_angle + 180 * (value / 100)
        
        # Draw the gauge
        wedge = Wedge((0, 0), r_outer, start_angle, end_angle, width=r_outer-r_inner, 
                      facecolor=color, edgecolor='white', linewidth=1)
        ax4.add_patch(wedge)
        
        # Add the value text
        ax4.text(pos, -0.2, f"{value}", ha='center', va='center', fontsize=12, 
                 color='white', fontweight='bold')
    
    # Set up the gauge chart
    ax4.set_xlim(-2, 2)
    ax4.set_ylim(-1.25, 1.25)
    ax4.set_aspect('equal')
    ax4.axis('off')
    ax4.set_facecolor('#0A1929')
    
    # Draw the gauge background (semi-circle)
    background = Wedge((0, 0), 0.8, 180, 360, width=0.2, facecolor='#333333', edgecolor='white', linewidth=1)
    ax4.add_patch(background)
    
    # Extract data for gauge chart
    bin_waste_types = bin_capacity_data['WasteType'].tolist()
    bin_utilization = bin_capacity_data['Utilization'].tolist()
    
    # Draw each gauge
    colors = ['#FF3366', '#33CC66', '#FFCC33', '#9966FF']
    positions = [-1.5, -0.5, 0.5, 1.5]
    
    for i, (pos, (waste_type, value)) in enumerate(zip(positions, zip(bin_waste_types, bin_utilization))):
        gauge(pos, value, colors[i], waste_type)
        ax4.text(pos, -0.4, waste_type, ha='center', va='center', fontsize=10, color=colors[i])
    
    ax4.set_title('Bin Capacity Utilization Chart', fontsize=16, pad=20)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=colors[i], markersize=10, 
                                 label=bin_waste_types[i])
                      for i in range(len(bin_waste_types))]
    ax4.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.6, 0.5), fontsize=12)
    
    # Adjust layout and save
    plt.tight_layout(pad=4.0)
    plt.subplots_adjust(top=0.92)
    #plt.suptitle('Waste Management Analytics Dashboard', fontsize=20, y=0.98)
    
    # Save the figure
    plt.savefig('waste_management_charts.png', dpi=300, bbox_inches='tight')
    
    print("Charts have been generated and saved as 'waste_management_charts.png'")
    
    # Display the figure
    plt.show()

if __name__ == "__main__":
    print("Waste Management Data Analysis")
    print("=" * 30)
    
    # Check if the Excel file exists, if not, provide instructions
    import os
    if not os.path.exists('waste_management_data.xlsx'):
        print("Error: 'waste_management_data.xlsx' not found.")
        print("Please run the JavaScript code to generate the Excel file first.")
    else:
        plot_all_charts()