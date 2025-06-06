#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import argparse
import seaborn as sns
import pandas as pd
from re import search, findall, MULTILINE
from os.path import basename, getsize
from typing import List, Any
from plotting import *


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
        'ukebpfjit': 'UniBPF',
        'linux': 'Linux',
        'uk': 'Unikraft',
        }

YLABEL = 'Throughput [Mpps]'
XLABEL = 'Packet size [B]'

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
                        default='throughput.pdf'
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
            dfs += [ arg_df ]
    df = pd.concat(dfs)

    df['pps'] = df['pps'].apply(lambda pps: pps / 1_000_000) # now mpps
    df['size'] = df['size'].astype(int)
    df['system'] = df['system'].apply(lambda row: system_map.get(str(row), row))
    df['vnf'] = df['vnf'].apply(lambda row: vnf_map.get(str(row), row))
    """
    columns = ['system', 'size', 'mpps', 'gbit']
    systems = [ "MorphOS", "MorphOS MPK",
               # "MorphOS MPK (perm)", "MorphOS MPK (copy)"
               ]
    sizes = [ 64, 1300, 1518 ]
    rows = []
    for system in systems:
        for size in sizes:
            value = 0
            if size == 64:
                match system:
                    case "MorphOS":
                        value = 2.4
                    case "MorphOS MPK":
                        value = 1.2
                    case "MorphOS MPK (perm)":
                        value = 1.2
                    case "MorphOS MPK (copy)":
                        value = 2.3
            else:
                match system:
                    case "MorphOS":
                        value = 10 * 1000 / 8 / (size + 7 + 1 + 12)
                    case "MorphOS MPK":
                        value = 10 * 1000 / 8 / (size + 7 + 1 + 12)
                    case "MorphOS MPK (perm)":
                        value = 10 * 1000 / 8 / (size + 7 + 1 + 12)
                    case "MorphOS MPK (copy)":
                        value = 9 * 1000 / 8 / (size + 7 + 1 + 12)

            gbit = value * (size + 7 + 1 + 12) * 8 / 1000

            rows += [[system, size, value, gbit]]

    # rows += [["MorphOS MPK", 600, 0, 5]]
    df = pd.DataFrame(rows, columns=columns)
    """


    # groups data, inserts new bars with 0 values between groups
    def group_data(df, grid_column_name, x_name, y_name, hue_name, hue_group_name, new_hue="grouped_hue"):
        dfs = []
        for grid_column in df[grid_column_name].unique():
            for x in df[x_name].unique():
                spacer = None
                for hue_group in df[hue_group_name].unique():
                    group_sample = None

                    # add spacer of previous group before adding this group
                    if spacer is not None:
                        dfs += [ spacer ]

                    # add this group
                    for hue in df[hue_name].unique():
                        # x, hue and sub_hue define a group
                        group_member = df[(df[grid_column_name] == grid_column) & (df[x_name] == x) & (df[hue_name] == hue) & (df[hue_group_name] == hue_group)].copy()
                        group_member[new_hue] = f"{hue_group} {hue}"
                        if len(group_member) > 0:
                            group_sample = group_member.loc[0].copy()

                        dfs += [ group_member ]

                    # create spacer for this group
                    if group_sample is not None:
                        spacer = group_sample
                        spacer[new_hue] = f"{hue_group} spacer"
                        spacer[y_name] = 0
        return pd.concat(dfs, ignore_index=True)

    # df = group_data(df, 'direction', 'size', 'pps', 'system', 'vnf', new_hue="grouped_system")

    # map colors to hues
    colors = sns.color_palette("pastel", len(df['system'].unique())-1) + [ mcolors.to_rgb('sandybrown') ]
    colors = sns.color_palette("pastel", 5) + [ mcolors.to_rgb('sandybrown') ]
    system_palette = dict(zip(df['system'].unique(), colors))
    # palette = dict(zip(df['grouped_system'].unique(), colors))

    # map all same systems to the same color
    # palette = dict()
    # for grouped_system in df['grouped_system'].unique():
    #     color = [v for k, v in system_palette.items() if k in grouped_system]
    #     if len(color) == 0:
    #         palette[grouped_system] = mcolors.to_rgb('black')
    #     else:
    #         palette[grouped_system] = color[0]


    def mpps_to_gbitps(mpps, size):
        return mpps * (size + 20) * 8 / 1000 # 20: preamble + packet gap

    def barplot_pointplot(*args, **kwargs):
        bar_kwargs = dict(kwargs.copy())
        del bar_kwargs['y_points']
        del bar_kwargs['color_by']
        del bar_kwargs['colors']
        # del bar_kwargs['hatch_by']
        # del bar_kwargs['hatches']
        ax1 = sns.barplot(*args, **bar_kwargs)
        # mybarplot.add_hatches(data=kwargs.get("data"), x=kwargs.get("x"), y=kwargs.get("y"), hue=kwargs.get("hue"), ax=ax1, hatch_by=kwargs.get("hatch_by"), hatches=kwargs.get("hatches"))
        # mybarplot.add_colors(data=kwargs.get("data"), x=kwargs.get("x"), y=kwargs.get("y"), hue=kwargs.get("hue"), ax=ax1, colors=kwargs.get("colors"), color_by=kwargs.get("color_by"))

        hues = [ 64, 128, 256, 512, 1024, 1280, 1518 ]

        ax2 = ax1.twinx().twiny()
        xlim = ax1.get_xlim()
        ax2.set_xticks([])
        ax2.set_xlim(left=xlim[0], right=xlim[1])
        ylim = ax1.get_ylim()
        # ax2.set_ylim(
        #     bottom=0,
        #     top=mpps_to_gbitps(ylim[1], max(hues))
        # )

        x = []
        y = []
        for bar in ax1.patches:
            x_coord = bar.get_bbox().x0 + bar.get_bbox().width / 2
            x_category_value = mybarplot.x_category_value(bar, ax1)
            y_val = mybarplot.y_value(bar)
            packet_size = int(x_category_value)
            gbitps = mpps_to_gbitps(y_val, packet_size)

            if y_val != 0:
                x += [ x_coord ]
                y += [ gbitps ]

        sns.scatterplot(*args,
                      x = x,
                      y = y,
                      legend=True,
                      # native_scale=True,
                      # linestyles='',
                      marker='v',
                      # palette="pastel",
                      c = [ "cornflowerblue" ],
                      s=20,
                      ax=ax2)

    # hatch_map = dict()
    # hatch_by = "vnf"
    # for hue_value, hatch in zip(df[hatch_by].unique(), HATCHES):
    #     hatch_map[hue_value] = hatch

    color_map = dict()
    color_by = "system"
    for hue_value, color in zip(df[color_by].unique(), colors):
        color_map[hue_value] = color

    barplot_pointplot(df,
               x='size',
               y='mpps',
               y_points='mpps',
               color_by=color_by,
               colors=color_map,
               # hatch_by=hatch_by,
               # hatches=hatch_map,
               hue='system',
               # palette=palette,
               edgecolor="dimgray",
               )

    def foo(plot_in_grid):
        plot_in_grid.twinx()
        pass

    # foo(grid.facet_axis(0, 0))

    def filter_legend(grid, keep_label):
        legend_data = dict()
        for label, handle in grid._legend_data.items():
            if keep_label(label):
                legend_data[label] = handle
        grid._legend_data = legend_data

    def legend_add_rectangle(grid, label, hatch=None, color="blue"):
        new_handle = Rectangle((0, 0), 1, 1, hatch=hatch, facecolor=color, edgecolor="dimgray", label=label)
        grid._legend_data[label] = new_handle

    def legend_add_line(grid, label, color="blue"):
        new_handle = Line2D([], [], color=color, marker='v', markersize=3, linestyle='', label=label)
        grid._legend_data[label] = new_handle


    # filter_legend(grid, lambda label: "spacer" not in label)
    #
    # grid._legend_data = dict()
    # # legend_add_line(grid, "", color="white")
    # for label, color in color_map.items():
    #     legend_add_rectangle(grid, label, color=color)
    # for label, hatch in hatch_map.items():
    #     legend_add_rectangle(grid, label, color="white", hatch=hatch)
    #
    #
    # grid.add_legend(
    #         bbox_to_anchor=(0.55, 0.3),
    #         loc='upper left',
    #         ncol=3, title=None, frameon=False,
    #                 )

    # # Fix the legend hatches
    # hatches = [ hatch for hatch in HATCHES for _ in range(3) ] # 3 legend items in each hue_group
    # for i, legend_patch in enumerate(grid._legend.get_patches()):
    #     hatch = hatches[i % len(hatches)]
    #     legend_patch.set_hatch(f"{hatch}{hatch}")
    # for i, legend_patch in enumerate(grid._legend.get_patches()):
    #     hatch = legend_patch.get_hatch()
    #     legend_patch.set_hatch(f"{hatch}{hatch}")


    # grid._legend_data = dict()
    # legend_add_rectangle(grid, "Throughput [Mpps]", color="white")
    # legend_add_line(grid, "Throughput [Gbit/s]", color="dimgray")
    #
    # grid.add_legend(
    #         bbox_to_anchor=(0.55, 0.3),
    #         loc='lower left',
    #         ncol=2, title=None, frameon=False,
    #                 )

    # def grid_set_titles(grid, titles):
    #     for ax, title in zip(grid.axes.flat, titles):
    #         ax.set_title(title)
    #
    # grid_set_titles(grid, ["Emulation and Mediation", "Passthrough"])
    #
    # grid.figure.set_size_inches(args.width, args.height)
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
    # grid.set_xlabels(XLABEL)
    # grid.set_ylabels(YLABEL)
    # def get_twin(grid, i, j, twin_nr=1):
    #     return grid.facet_axis(i, j)._twinned_axes.get_siblings(grid.facet_axis(i, j))[twin_nr]
    # get_twin(grid, 0, 1).set_ylabel("Throughput [Gbit/s]")
    # get_twin(grid, 0, 2).set_ylabel("Throughput [Gbit/s]")
    #
    # grid.facet_axis(0, 0).annotate(
    #     "↑ Higher is better", # or ↓ ← ↑ →
    #     xycoords="axes points",
    #     # xy=(0, 0),
    #     xy=(0, 0),
    #     xytext=(-37, -28),
    #     # fontsize=FONT_SIZE,
    #     color="navy",
    #     weight="bold",
    # )

    # plt.xlabel(XLABEL)
    # plt.ylabel(YLABEL)

    # plt.ylim(0, 1)
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
    # plt.tight_layout(rect=(0, 0, 1, 1))
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(right=0.95, wspace=0.15, hspace=0.5)
    plt.savefig(args.output.name)
    plt.close()





if __name__ == '__main__':
    main()
