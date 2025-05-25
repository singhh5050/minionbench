#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def main():
    # Ensure the visualizations directory exists
    os.makedirs("visualizations", exist_ok=True)

    # Load the dataset
    df = pd.read_csv("/Users/harshsingh/Desktop/minionbench/results/metrics.csv")

    # Derived metrics
    df['throughput'] = df['generated_tokens'] / df['latency']
    df['tok_per_joule'] = df['generated_tokens'] / df['energy_j']
    df['length_bucket'] = pd.qcut(df['input_tokens'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    # 1. Barplots of averages by protocol
    metrics = (
        df.groupby('protocol')
          .agg({
              'generated_tokens': 'mean',
              'latency': 'mean',
              'throughput': 'mean',
              'tok_per_joule': 'mean'
          })
          .reset_index()
    )

    # 2. Heatmap: latency by category and protocol
    pivot = df.pivot_table(
        index='category',
        columns='protocol',
        values='latency',
        aggfunc='mean'
    )

    # Format the numbers for annotation
    annot_data = pivot.round(1).astype(str)

    # Plot the heatmap normally — NO mask needed
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        pivot,
        annot=annot_data,
        fmt='',              # Empty fmt because annot_data is already string
        cmap='rocket',
        cbar=True,
        linewidths=0.5,
        linecolor='gray',
        square=False
    )
    plt.title('Average Latency (s) by Category and Protocol')
    plt.xlabel('Protocol')
    plt.ylabel('Economic Category')
    plt.tight_layout()
    plt.savefig("visualizations/heatmap_latency_by_category_and_protocol.png")
    plt.close()



    # 3. Distribution of generated tokens by protocol
    plt.figure()
    sns.boxplot(data=df, x='protocol', y='generated_tokens')
    plt.title('Distribution of Generated Tokens by Protocol')
    plt.xlabel('Protocol')
    plt.ylabel('Tokens')
    plt.tight_layout()
    plt.savefig("visualizations/distribution_generated_tokens_by_protocol.png")
    plt.close()

    # 4. Scatter: energy vs latency
    plt.figure()
    ax = sns.scatterplot(
        data=df, x='tok_per_joule', y='throughput',
        hue='protocol', size='output_tokens',
        sizes=(20, 200), alpha=0.7
    )
    plt.title('Energy Efficiency vs. Generation Speed by Protocol')
    plt.xlabel('Tokens per Joule')
    plt.ylabel('Tokens per Second')
    plt.tight_layout()
    # Move the legend to the right
    ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
    plt.savefig("visualizations/efficiency_vs_speed_by_protocol.png", bbox_inches='tight')
    plt.close()

    # 5. Latency over run order
    df_sorted = df.reset_index()
    plt.figure()
    sns.lineplot(
        data=df_sorted, x='index', y='latency', hue='protocol'
    )
    plt.title('Latency over Run Order')
    plt.xlabel('Run Index')
    plt.ylabel('Latency (Seconds)')
    plt.tight_layout()
    plt.savefig("visualizations/latency_over_run_order.png")
    plt.close()

    # 6. Latency by input length bucket and protocol
    g = sns.catplot(
        data=df, x='length_bucket', y='latency',
        hue='protocol', kind='bar'
    )
    g.fig.suptitle('Average Latency by Input Token Quartile and Protocol')
    g.set_axis_labels('Input Token Quartile', 'Latency (Seconds)')
    g.tight_layout()
    # Move the legend to the right
    g._legend.set_bbox_to_anchor((1.05, 0.5))  # (x, y) - x > 1 moves it to the right
    g._legend.set_loc('center right')
    g.savefig("visualizations/latency_by_input_token_quartile_and_protocol.png")
    plt.close()

        # 7. Violin: tokens per joule distribution by protocol
    plt.figure()
    sns.violinplot(data=df, x='protocol', y='tok_per_joule', inner='quartile')
    plt.title('Distribution of Tokens per Joule by Protocol')
    plt.xlabel('Protocol')
    plt.ylabel('Tokens per Joule')
    plt.tight_layout()
    plt.savefig("visualizations/violin_tokens_per_joule_by_protocol.png")
    plt.close()

    # 8. Enhanced boxplot: energy consumption distribution by category with protocol overlay
    plt.figure(figsize=(10, 6))
    # Gray base boxes
    sns.boxplot(data=df, y='category', x='energy_j', orient='h', color='lightgray', showfliers=False)
    # Colored strip of points per protocol
    sns.stripplot(data=df, y='category', x='energy_j', hue='protocol', orient='h', dodge=True, size=4, alpha=0.7)
    plt.title('Energy Consumption by Category (with Protocol Overlay)')
    plt.xlabel('Energy (Joules)')
    plt.ylabel('Economic Category')
    plt.legend(title='Protocol', bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig("visualizations/energy_by_category_protocol_overlay.png")
    plt.close()

    # 9. Minion: Generated vs. Output Tokens by Category (with ratio annotation)
    df_minion = df[df['protocol'] == 'minion']
    summary = (
        df_minion
        .groupby('category')
        .agg(
            generated_tokens=('generated_tokens', 'mean'),
            output_tokens=('output_tokens', 'mean')
        )
        .reset_index()
    )
    summary['ratio'] = summary['generated_tokens'] / summary['output_tokens']
    melt_summary = summary.melt(id_vars='category', var_name='metric', value_name='tokens')

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=melt_summary,
        y='category', x='tokens',
        hue='metric', palette='Set2',
        ci=None,
        order=summary['category']  # ensure correct category order
    )
    # Annotate ratio next to each bar group
    for idx, row in summary.iterrows():
        max_val = max(row['generated_tokens'], row['output_tokens'])
        ax.text(
            max_val * 1.02, idx,
            f"{row['ratio']:.2f}×",
            va='center'
        )
    plt.title('Minion: Generated vs. Output Tokens by Category')
    plt.xlabel('Token Count')
    plt.ylabel('Economic Category')
    plt.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig("visualizations/minion_generated_vs_output_by_category.png")
    plt.close()



if __name__ == '__main__':
    main()