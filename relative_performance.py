#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import seaborn as sns
import pandas as pd
from re import search, findall, MULTILINE
from os.path import basename, getsize
from typing import List, Any
import plotting
import latency_cdf

hatches = plotting.HATCHES


COLORS = [ str(i) for i in range(20) ]
# COLORS = mcolors.CSS4_COLORS.keys()
# COLORS = [
#     'blue',
#     'cyan',
#     'green',
#     'yellow',
#     'orange',
#     'red',
#     'magenta',
# ]

# hue_map = {
#     '9_vmux-dpdk-e810_hardware': 'vmux-emu (w/ rte_flow)',
#     '9_vmux-med_hardware': 'vmux-med (w/ rte_flow)',
#     '9_vmux-dpdk-e810_software': 'vmux-emu',
#     '9_vmux-med_software': 'vmux-med',
#     '1_vfio_software': 'qemu-pt',
#     '1_vmux-pt_software': 'vmux-pt',
#     '1_vmux-pt_hardware': 'vmux-pt (w/ rte_flow)',
#     '1_vfio_hardware': 'qemu-pt (w/ rte_flow)',
# }

system_map = {
        'ebpf-click-unikraftvm': 'Uk click (eBPF)',
        'click-unikraftvm': 'Uk click',
        'click-linuxvm': 'Linux click',
        }

grid_title_map = {
    'direction = latency': 'Latency',
}

YLABEL = 'Speedup'
XLABEL = 'VNF'

def map_hue(df_hue, hue_map):
    return df_hue.apply(lambda row: hue_map.get(str(row), row))



def setup_parser():
    parser = argparse.ArgumentParser(
        description='Plot packet loss graph'
    )

    parser.add_argument('-t',
                        '--title',
                        type=str,
                        help='Title of the plot',
                        )
    parser.add_argument('-W', '--width',
                        type=float,
                        default=12,
                        help='Width of the plot in inches'
                        )
    parser.add_argument('-H', '--height',
                        type=float,
                        default=6,
                        help='Height of the plot in inches'
                        )
    parser.add_argument('-o', '--output',
                        type=argparse.FileType('w+'),
                        help='''Path to the output plot
                             (default: packet_loss.pdf)''',
                        default='relative_performance.pdf'
                        )
    parser.add_argument('-l', '--logarithmic',
                        action='store_true',
                        help='Plot logarithmic latency axis',
                        )
    parser.add_argument('-s', '--slides',
                        action='store_true',
                        help='Use other setting to plot for presentation slides',
                        )
    for color in COLORS:
        parser.add_argument(f'--{color}',
                            type=argparse.FileType('r'),
                            nargs='+',
                            help=f'''Paths to MoonGen measurement logs for
                                  the {color} plot''',
                            )
    for color in COLORS:
        parser.add_argument(f'--{color}-name',
                            type=str,
                            default=color,
                            help=f'''Name of {color} plot''',
                            )

    parser.add_argument(f'--uk-histogram',
                        type=argparse.FileType('r'),
                        nargs='+',
                        help=f'''Paths to UK histogram CSVs''',
                        )
    parser.add_argument(f'--linux-histogram',
                        type=argparse.FileType('r'),
                        nargs='+',
                        help=f'''Paths to UK histogram CSVs''',
                        )

    return parser


def parse_args(parser):
    args = parser.parse_args()

    if not any([args.__dict__[color] for color in COLORS]):
        parser.error('At least one set of moongen log paths must be ' +
                     'provided')

    return args

# hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O']
hatches_used = 0

# Define a custom function to add hatches to the bar plots
def barplot_with_hatches(*args, **kwargs):
    global hatches_used
    sns.barplot(*args, **kwargs)
    for i, bar in enumerate(plt.gca().patches):
        hatch = hatches[hatches_used % len(hatches)]
        print(hatch)
        bar.set_hatch(hatch)
        hatches_used += 1


def main():
    parser = setup_parser()
    args = parse_args(parser)

    # fig = plt.figure(figsize=(args.width, args.height))
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_axisbelow(True)
    if args.title:
        plt.title(args.title)
    plt.grid()
    # plt.xlim(0, 0.83)
    log_scale = (False, True) if args.logarithmic else False
    # ax.set_yscale('log' if args.logarithmic else 'linear')

    dfs = []
    for color in COLORS:
        if args.__dict__[color]:
            arg_dfs = [ pd.read_csv(f.name) for f in args.__dict__[color] ]
            arg_df = pd.concat(arg_dfs)
            name = args.__dict__[f'{color}_name']
            # arg_df["hue"] = name
            dfs += [ arg_df ]
            # throughput = ThroughputDatapoint(
            #     moongen_log_filepaths=[f.name for f in args.__dict__[color]],
            #     name=args.__dict__[f'{color}_name'],
            #     color=color,
            # )
            # dfs += color_dfs
    df = pd.concat(dfs)
    # hue = ['repetitions', 'num_vms', 'interface', 'fastclick']
    # groups = df.groupby(hue)
    # summary = df.groupby(hue)['rxMppsCalc'].describe()
    # df_hue = df.apply(lambda row: '_'.join(str(row[col]) for col in ['repetitions', 'interface', 'fastclick', 'rate']), axis=1)
    # df_hue = map_hue(df_hue, hue_map)
    # df['is_passthrough'] = df.apply(lambda row: True if "vmux-pt" in row['interface'] or "vfio" in row['interface'] else False, axis=1)

    linux_histogram = latency_cdf.LatencyHistogram(args.linux_histogram[0].name)
    uk_histogram = latency_cdf.LatencyHistogram(args.uk_histogram[0].name)

    df = df[df['size'] == 64]
    df.loc[(df["vnf"] == "mirror") | (df["vnf"] == "nat"), "direction"] = "bi"
    df['pps'] = df['pps'].apply(lambda pps: pps / 1_000_000) # now mpps
    dfa = df[(df['size'] == 64) & (df['direction'] == 'rx') & (df['vnf'] == 'empty')]
    # b = a[a['system'] == 'uk'] / a[a['system'] == 'linux']


    columns = ['vnf', 'direction', 'pps']
    vnfs = [ "empty", "filter", "ids", "mirror", "nat", "latency" ]
    rows = []
    for direction in ["rx", "tx", "bi", "latency" ]:
        for vnf in vnfs:
            df_ = df[(df['vnf'] == vnf) & (df['direction'] == direction)]
            value = None
            def speedup():
                old = df_[df_['system'] == 'linux']['pps'].mean()
                new = df_[df_['system'] == 'uk']['pps'].mean()
                return new / old
            match (vnf, direction):
                case ("empty", "rx"):
                    value = speedup()
                case ("filter", "rx"):
                    value = speedup()
                case ("ids", "rx"):
                    value = speedup()

                case ("empty", "tx"):
                    value = speedup()
                case ("filter", "tx"):
                    value = speedup()
                case ("ids", "tx"):
                    value = speedup()

                case ("mirror", "bi"):
                    value = speedup()
                case ("nat", "bi"):
                    value = speedup()

                case ("mirror", "latency"):
                    value = linux_histogram._percentile50 / uk_histogram._percentile50
                case ("nat", "latency"):
                    value = 0

            # if direction = "tx":
            #     value = 1
            # # if system == "click-unikraftvm":
            # #     value = 11
            # # if system == "click-linuxvm":
            # #     value = 9
            # if vnf == "none" and direction == "rx":
            #     value = 3
            # if vnf == "filter" and direction == "rx":
            #     value = 2.5
            # if vnf == "dpi" and direction == "rx":
            #     value = 2
            # if vnf == "nat" and direction == "rx":
            #     value = 2
            # if vnf == "latency" and direction == "rx":
            #     value = 1.5


            if value is not None:
                rows += [[vnf, direction, value]]
    df = pd.DataFrame(rows, columns=columns)

    # df['system'] = df['system'].apply(lambda row: system_map.get(str(row), row))

    # map colors to hues
    # colors = sns.color_palette("pastel", len(df['system'].unique())-1) + [ mcolors.to_rgb('sandybrown') ]
    # palette = dict(zip(df['system'].unique(), colors))

    # Plot using Seaborn
    grid = sns.FacetGrid(df,
            col='direction',
            sharey = True,
            sharex = False,
            # col_wrap=2,
            gridspec_kws={"width_ratios": [3, 3, 2, 2]},
    )
    grid.map_dataframe(sns.barplot,
               x='vnf',
               y='pps',
               # hue='system',
               # palette=palette,
               edgecolor="dimgray",
               )

    # Add horizontal line at y=1 to each subplot
    def add_hline(**kwargs):
        plt.axhline(y=1, color='darkgray', linestyle='--', linewidth=1)

    grid.map(add_hline)

    # grid.add_legend(
    #         # bbox_to_anchor=(0.5, 0.77),
    #         loc='right',
    #         ncol=1, title=None, frameon=False,
    #                 )
    #

    # def grid_set_titles(grid, titles):
    #     for ax, title in zip(grid.axes.flat, titles):
    #         ax.set_title(title)
    #
    # grid_set_titles(grid, ["Emulation and Mediation", "Passthrough"])
    #
    plotting.map_grid_titles(grid, grid_title_map)
    grid.figure.set_size_inches(args.width, args.height)
    # grid.set_titles("foobar")
    # plt.subplots_adjust(left=0.06)
    # bar = sns.barplot(x='num_vms', y='rxMppsCalc', hue="hue", data=pd.concat(dfs),
    #             palette='colorblind',
    #             edgecolor='dimgray',
    #             # kind='bar',
    #             # capsize=.05,  # errorbar='sd'
    #             # log_scale=log_scale,
    #             ax=ax,
    #             )
    # sns.move_legend(
    #     ax, "lower right",
    #     # bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
    # )
    #
    # sns.move_legend(
    #     grid, "lower center",
    #     bbox_to_anchor=(0.45, 1),
    #     ncol=1,
    #     title=None,
    #     # frameon=False,
    # )
    grid.set_xlabels(XLABEL)
    grid.set_ylabels(YLABEL)
    #
    grid.facet_axis(0, 0).annotate(
        "↑ Higher is better", # or ↓ ← ↑ →
        xycoords="axes points",
        # xy=(0, 0),
        xy=(0, 0),
        xytext=(-37, -40),
        # fontsize=FONT_SIZE,
        color="navy",
        weight="bold",
    )

    # plt.xlabel(XLABEL)
    # plt.ylabel(YLABEL)

    plt.ylim(0, 4.5)
    if not args.logarithmic:
        plt.ylim(bottom=0)
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.0f')

    # # iterate through each container, hatch, and legend handle
    # for container, hatch, handle in zip(ax.containers, hatches, ax.get_legend().legend_handles[::-1]):
    #     # update the hatching in the legend handle
    #     handle.set_hatch(hatch)
    #     # iterate through each rectangle in the container
    #     for rectangle in container:
    #         # set the rectangle hatch
    #         rectangle.set_hatch(hatch)

    # # Loop over the bars
    # for i,thisbar in enumerate(bar.patches):
    #     # Set a different hatch for each bar
    #     thisbar.set_hatch(hatches[i % len(hatches)])

    # legend = plt.legend()
    # legend.get_frame().set_facecolor('white')
    # legend.get_frame().set_alpha(0.8)
    # fig.tight_layout(rect = (0, 0, 0, 0.1))
    # ax.set_position((0.1, 0.1, 0.5, 0.8))
    # plt.tight_layout(pad=0.1)
    # plt.subplots_adjust(right=0.78)
    # fig.tight_layout(rect=(0, 0, 0.3, 1))
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(args.output.name)
    plt.close()





if __name__ == '__main__':
    main()
