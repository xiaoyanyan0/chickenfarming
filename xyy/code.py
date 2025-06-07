import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm # For colormaps
import re # For parsing bin strings

def _parse_and_calculate_bin_properties(df_input):
    """
    Parses bin strings and calculates properties for plotting.
    Expected df_input columns: 'Bin'
    Returns a DataFrame with: 'Parsed_Lower', 'Parsed_Upper', 
                             'Bar_X_Lower', 'Bar_Width', 'Line_X_Center',
                             and a list of x_tick_values.
    """
    df = df_input.copy()

    def parse_value(val_str):
        if val_str == '-inf':
            return -np.inf # Changed from np.NINF
        elif val_str == 'inf':
            return np.inf
        else:
            return float(val_str)

    # Regex to extract bounds: captures open_bracket, lower_bound, upper_bound, close_bracket
    pattern = re.compile(r"^\s*([(\[])\s*(-inf|[\d\.\-]+)\s*,\s*(inf|[\d\.\-]+)\s*([)\]])\s*$")
    
    parsed_bounds = df['Bin'].apply(lambda x: pattern.match(x).groups() if pattern.match(x) else (None, np.nan, np.nan, None))
    
    if parsed_bounds.apply(lambda x: x is None).any():
        raise ValueError("One or more 'Bin' strings could not be parsed. Ensure format is like '(-inf, 10.0)' or '[10.0, 20.0)'.")

    df['Parsed_Lower_Str'] = parsed_bounds.apply(lambda x: x[1])
    df['Parsed_Upper_Str'] = parsed_bounds.apply(lambda x: x[2])

    df['Parsed_Lower'] = df['Parsed_Lower_Str'].apply(parse_value)
    df['Parsed_Upper'] = df['Parsed_Upper_Str'].apply(parse_value)

    # Calculate initial widths (can be inf)
    df['Width_Raw'] = df['Parsed_Upper'] - df['Parsed_Lower']

    # Calculate Bar_Width, handling infinite intervals
    bar_widths = []
    if len(df) == 0: # Handle empty dataframe
        return df, []

    for i in range(len(df)):
        current_parsed_lower = df['Parsed_Lower'].iloc[i]
        current_parsed_upper = df['Parsed_Upper'].iloc[i]
        
        if current_parsed_lower == -np.inf: # First bin is (-inf, val)
            if len(df) > 1: # More than one bin
                next_bin_lower = df['Parsed_Lower'].iloc[i+1]
                next_bin_upper = df['Parsed_Upper'].iloc[i+1]
                width = next_bin_upper - next_bin_lower
                if np.isinf(width) or pd.isna(width): # If next bin is also infinite or undefined width
                    width = 2.0 # Default width
                    if not np.isinf(current_parsed_upper): # Try to make it sensible if upper bound is finite
                         width = current_parsed_upper - (current_parsed_upper - 2.0) # Default to width of 2 from upper
                    else: # both current upper and next width are inf
                         width = 2.0 
            else: # Only one bin, (-inf, val)
                width = 2.0 
                if not np.isinf(current_parsed_upper):
                    width = current_parsed_upper - (current_parsed_upper - 2.0)
            bar_widths.append(width)
        elif current_parsed_upper == np.inf: # Last bin is (val, inf)
            if len(df) > 1: # More than one bin
                prev_bin_lower = df['Parsed_Lower'].iloc[i-1]
                prev_bin_upper = df['Parsed_Upper'].iloc[i-1]
                width = prev_bin_upper - prev_bin_lower
                if np.isinf(width) or pd.isna(width): # If previous bin was also infinite or undefined width
                    width = 2.0 # Default width
                    if not np.isinf(current_parsed_lower): # Try to make it sensible
                        width = (current_parsed_lower + 2.0) - current_parsed_lower
                    else:
                        width = 2.0
            else: # Only one bin, (val, inf)
                width = 2.0
                if not np.isinf(current_parsed_lower):
                    width = (current_parsed_lower + 2.0) - current_parsed_lower
            bar_widths.append(width)
        else: # Finite interval
            width = df['Width_Raw'].iloc[i]
            if np.isinf(width) or pd.isna(width): # Should not happen for finite interval unless bounds are same
                width = 0.1 # very small width for degenerate finite interval
            bar_widths.append(width)
            
    df['Bar_Width'] = bar_widths
    df.loc[df['Bar_Width'] <= 0, 'Bar_Width'] = 0.1 # Ensure positive width

    # Calculate Bar_X_Lower (left edge of the bar)
    bar_x_lower = []
    for i in range(len(df)):
        if df['Parsed_Lower'].iloc[i] == -np.inf:
            bar_x_lower.append(df['Parsed_Upper'].iloc[i] - df['Bar_Width'].iloc[i])
        else:
            bar_x_lower.append(df['Parsed_Lower'].iloc[i])
    df['Bar_X_Lower'] = bar_x_lower

    # Calculate Line_X_Center (midpoint of the bar for the line plot)
    df['Line_X_Center'] = df['Bar_X_Lower'] + df['Bar_Width'] / 2

    # Determine x-tick values for the plot
    x_tick_values = []
    if not df.empty:
        x_tick_values = sorted(list(set(df['Bar_X_Lower'].tolist() + \
                                   [(df['Bar_X_Lower'].iloc[-1] + df['Bar_Width'].iloc[-1])])))
    
    return df, x_tick_values


def plot_temperature_mortality_chart(dataframe):
    """
    Generates a combined bar and line chart from the provided DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame with columns 'Bin', 'Count', and 'MORTALITY_RATE'.
                                  'Bin' format: '(-inf, 10.13)', '[10.13, 12.55)', etc.
    """
    
    plt.rcParams['font.sans-serif'] = [
        'SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 
        'PingFang SC', 'Source Han Sans CN', 'Noto Sans CJK SC', 
        'Arial Unicode MS'
    ]
    plt.rcParams['axes.unicode_minus'] = False

    try:
        df_processed, x_tick_values = _parse_and_calculate_bin_properties(dataframe)
        if df_processed.empty:
            print("Input DataFrame is empty or could not be processed. No chart generated.")
            return
    except Exception as e:
        print(f"Error processing bin data: {e}")
        print("Please ensure the 'Bin' column format is as expected, e.g., '(-inf, 10.13)' or '[10.13, 12.55)'.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 8))

    num_bars = len(df_processed)
    color_map_bars = cm.get_cmap('Blues')
    bar_colors = [color_map_bars(i) for i in np.linspace(0.3, 0.9, num_bars)]

    bars = ax1.bar(df_processed['Bar_X_Lower'], df_processed['Count'], 
                   width=df_processed['Bar_Width'], color=bar_colors, 
                   alpha=0.8, align='edge', label='数量 (Count)')

    ax1.set_xlabel('温度 (Temperature)', fontsize=12)
    ax1.set_ylabel('数量 (Count)', color='dimgray', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='dimgray', labelsize=10)
    ax1.tick_params(axis='x', labelsize=10)

    ax2 = ax1.twinx()
    color_line = 'crimson'
    line_plot = ax2.plot(df_processed['Line_X_Center'], df_processed['MORTALITY_RATE'], 
                         color=color_line, marker='o', linestyle='-', 
                         linewidth=2, markersize=7, label='MORTALITY_RATE')
    ax2.set_ylabel('MORTALITY_RATE', color=color_line, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_line, labelsize=10)

    for i in range(len(df_processed)):
        ax2.text(df_processed['Line_X_Center'].iloc[i], 
                 df_processed['MORTALITY_RATE'].iloc[i] + 0.001, 
                 f"{df_processed['MORTALITY_RATE'].iloc[i]:.4f}",
                 ha='center', va='bottom', fontsize=9, color=color_line, fontweight='medium')

    if x_tick_values:
        ax1.set_xticks(x_tick_values)
        ax1.set_xticklabels([f"{val:.2f}" for val in x_tick_values], rotation=45, ha="right")
        min_width_for_padding = df_processed['Bar_Width'].replace(0, np.nan).min() # Avoid 0 width for padding
        if pd.isna(min_width_for_padding) or min_width_for_padding == 0: min_width_for_padding = 1.0

        ax1.set_xlim(x_tick_values[0] - 0.5 * min_width_for_padding, 
                     x_tick_values[-1] + 0.5 * min_width_for_padding)
    else: # No ticks if df was empty
        ax1.set_xticks([])


    plt.title('数量和MORTALITY_RATE随温度变化图', fontsize=16)
    
    handles = [bars] + line_plot
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc='upper right', fontsize=10)

    fig.tight_layout()
    plt.savefig("temperature_mortality_chart_refactored.png")
    plt.show()
    
    print("\nRefactored chart generated and saved as temperature_mortality_chart_refactored.png")
    # print("\nProcessed DataFrame for plotting:")
    # print(df_processed[['Bin', 'Parsed_Lower', 'Parsed_Upper', 'Bar_X_Lower', 'Bar_Width', 'Line_X_Center', 'Count', 'MORTALITY_RATE']].to_string())


# --- Example Usage ---
data_for_df = {
    'Bin': ['(-inf, 10.13)', '[10.13, 12.55)', '[12.55, 15.02)', '[15.02, 15.94)', '[15.94, inf)'],
    'Count': [210, 53, 41, 21, 79],
    'Count (%)': [0.519802, 0.131188, 0.101485, 0.051980, 0.195545],
    'Event rate': [0.276190, 0.150943, 0.121951, 0.047619, 0.113924],
    'MORTALITY_RATE': [0.100248, 0.085784, 0.076452, 0.067034, 0.078049]
}
sample_input_df = pd.DataFrame(data_for_df)

plot_temperature_mortality_chart(sample_input_df)
