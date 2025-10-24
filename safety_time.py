#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import seaborn as sns
import pandas as pd
from re import search, findall, MULTILINE
from os.path import basename, getsize, isfile, join, dirname
from typing import List, Any
from plotting import HATCHES as _hatches
from tqdm import tqdm

from reconfiguration_vnfs import parse_data, log


# resource on how to do stackplots:
# https://stackoverflow.com/questions/59038979/stacked-bar-chart-in-seaborn

hatches = _hatches.copy()
hatches[0] = _hatches[6]
hatches[6] = _hatches[2]
hatches[2] = _hatches[0]


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
        'ebpf-click-unikraftvm': 'Unikraft click (eBPF)',
        'click-unikraftvm': 'Unikraft click',
        'click-linuxvm': 'Linux click',
        }

YLABEL = 'Time [ms]'
X1LABEL = ''
X2LABEL = 'Replacement frequency'

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
                             (default: safety_time.pdf)''',
                        default='safety_time.pdf'
                        )
    parser.add_argument('-l', '--logarithmic',
                        action='store_true',
                        help='Plot logarithmic latency axis',
                        )
    parser.add_argument('-c', '--cached',
                        action='store_true',
                        help='Use cached version of parsed data',
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
    parser.add_argument(f'--bpfbuild',
                        type=argparse.FileType('r'),
                        nargs='+',
                        help=f'''Paths to UK bpfbuild CSVs''',
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

    if args.cached and isfile("/tmp/reconfiguration_vnf.pkl"):
        log("Using cached data")
        df = pd.read_pickle("/tmp/reconfiguration_vnf.pkl")
    else:
        dfs = []
        for color in COLORS:
            if args.__dict__[color]:
                log(f"Reading files for --name-{color}")
                arg_dfs = [ pd.read_csv(f.name) for f in tqdm(args.__dict__[color]) ]
                arg_df = pd.concat(arg_dfs)
                name = args.__dict__[f'{color}_name']
                arg_df["arglabel"] = name
                dfs += [ arg_df ]
                # throughput = ThroughputDatapoint(
                #     moongen_log_filepaths=[f.name for f in args.__dict__[color]],
                #     name=args.__dict__[f'{color}_name'],
                #     color=color,
                # )
                # dfs += color_dfs
        df = pd.concat(dfs, ignore_index=True)
        # hue = ['repetitions', 'num_vms', 'interface', 'fastclick']
        # groups = df.groupby(hue)
        # summary = df.groupby(hue)['rxMppsCalc'].describe()
        # df_hue = df.apply(lambda row: '_'.join(str(row[col]) for col in ['repetitions', 'interface', 'fastclick', 'rate']), axis=1)
        # df_hue = map_hue(df_hue, hue_map)
        # df['is_passthrough'] = df.apply(lambda row: True if "vmux-pt" in row['interface'] or "vfio" in row['interface'] else False, axis=1)

        df = parse_data(df)
        df.to_pickle("/tmp/reconfiguration_vnf.pkl")

    df_full = df.copy()
    df['msec'] = df['nsec'].apply(lambda nsec: nsec / 1_000_000.0)
    nat_mean_msecs = df[(df["system"]=="ukebpfjit")&(df["vnf"]=="nat")].groupby("label")["msec"].mean().reset_index()
    nat_mean_msecs=nat_mean_msecs.assign(system='\nReconfiguration')
    nat_mean_msecs = nat_mean_msecs.rename(columns={'system': 'system', 'label': 'contributor', 'msec': 'restart_s'})
    nat_mean_msecs.loc[nat_mean_msecs['contributor'] == 'other', 'contributor'] = 'Control'
    oob_df = pd.read_csv(args.bpfbuild[0].name)
    oob_df=oob_df.assign(system='Out-of-band\n')

    # fig = plt.figure(figsize=(args.width, args.height))
    # Create a figure with two subplots side by side, sharing y axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(args.width, args.height), sharey=True)

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_axisbelow(True)
    if args.title:
        plt.title(args.title)
    # plt.grid()
    # plt.xlim(0, 0.83)
    log_scale = (False, True) if args.logarithmic else False
    # ax.set_yscale('log' if args.logarithmic else 'linear')

    columns = ['system', 'contributor', 'restart_s']
    systems = [ "Out-of-band\n", "\nReconfiguration" ]
    contributors = [ "Compile", "Link", "Verify", "Load", "Validate", "JIT", "Control" ]

    df = pd.concat([nat_mean_msecs[nat_mean_msecs['contributor'].isin(contributors)], oob_df])

    df['system'] = df['system'].apply(lambda row: system_map.get(str(row), row))
    df['system'] = pd.Categorical(df['system'], systems)

    df = df[df['contributor'] != 'Load'] # not visible anyways. Remove so that it won't be visible in Legend


    # map colors to hues
    # colors = sns.color_palette("pastel", len(df['hue'].unique())-1) + [ mcolors.to_rgb('sandybrown') ]
    # palette = dict(zip(df['hue'].unique(), colors))

    # Plot using Seaborn
    sns.histplot(
               data=df,
               ax=ax1,
               x='system',
               weights='restart_s',
               hue="contributor",
               hue_order = ['Compile', 'Link', 'Verify', 'Validate', 'JIT', 'Control'],
               multiple="stack",
               # palette=palette,
               edgecolor="dimgray",
               shrink=0.8,
               )
    ax1.set_title('(a) Safety overhead  ')


    avg_compile = df[df["contributor"] == "Compile"]["restart_s"].mean() # actually msec not sec
    avg_link = df[df["contributor"] == "Link"]["restart_s"].mean()
    avg_verify = df[df["contributor"] == "Verify"]["restart_s"].mean()
    # avg_load = df[df["contributor"] == "Load"]["restart_s"].mean()
    avg_validate = df[df["contributor"] == "Validate"]["restart_s"].mean()
    avg_jit = df[df["contributor"] == "JIT"]["restart_s"].mean()
    avg_control = df[df["contributor"] == "Control"]["restart_s"].mean()

    total_out_of_band = avg_compile + avg_link + avg_verify
    total_reconfiguration = avg_validate + avg_jit + avg_control


    # Create sample data
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)
    x = [ 10, 5, 1 ]
    y = [ 7, 2, 1 ]

    # We define the amortized out-of-band overhead with a update frequency $f_u$, and a reconfiguration frequency $f_r$ given a verification time $t_v$ as $t_a = t_v * f_u / f_r$.
    HUE_LABEL = "Update frequency"
    columns = ['t_v', 'f_u', HUE_LABEL, 'f_r', 't_a']
    t_v =  total_out_of_band # approx. 335 # in milliseconds
    secondly = 1 # 1/s
    minutely = 1/60 # 1/min
    hourly= 1/(60*60) # 1/h
    daily = 1/(60*60*24) # 1/day
    monthly = 1/(60*60*24*30) # 1/month
    # f_rs = [ secondly, minutely, hourly, daily, monthly ]
    # f_rs = np.logspace( monthly, minutely, num=4, endpoint=True)
    # f_rs = np.outer(f_rs, np.arange(1,10,1)).flatten()
    f_us = [ (monthly, '1/month'), (hourly, '1/day') ]
    f_rs = [ 1.0/pow(2, i) for i in range(1, 30) ] + [ f_u for f_u, _ in f_us ]
    rows = []
    for f_u, hue in f_us:
        for f_r in f_rs:
            if f_u <= f_r: # reconfiguration frequency must be higher than update frequency
                t_a = t_v * f_u / f_r
                # t_a = t_v * (1/f_r) / (1/f_u)
                rows += [[t_v, f_u, hue, f_r, t_a]]
    df = pd.DataFrame(rows, columns=columns)
    print(df)

    # Create line plot
    sns.lineplot(df, x="f_r", y="t_a", hue=HUE_LABEL, ax=ax2)
    ax2.set_title('(b) Overhead amortization  ')
    ax2.set_xscale('log')

    # set ticks
    f_rs = [ secondly, minutely, hourly, daily, monthly ]
    ax2.set_xticks(f_rs)
    ax2.set_xticklabels(['1/s', '', '1/hour', '','  1/month'])

    # Remove y-axis label from the second plot since it's shared
    ax2.set_ylabel('$t_a$ [s]')

    # sns.add_legend(
    #         # bbox_to_anchor=(0.5, 0.77),
    #         loc='right',
    #         ncol=1, title=None, frameon=False,
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

    # remove red "Load" from legend
    sns.move_legend(
        ax1, "upper right",
        bbox_to_anchor=(1.17, -0.43), ncol=2, title=None, frameon=True,
    )
    sns.move_legend(
        ax2, "upper left",
        bbox_to_anchor=(0.1, -0.43), ncol=1, frameon=True,
    )
    # grid.add_legend(
    #     # bbox_to_anchor=(0.5, 0.77),
    #     loc='upper center',
    #     ncol=6 if not args.slides else 5,
    #     title=None, frameon=False,
    #             )

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
    #
    ax1.annotate(
        "↓ Lower is better", # or ↓ ← ↑ →
        xycoords="axes points",
        # xy=(0, 0),
        xy=(0, 0),
        xytext=(-45, -30),
        # fontsize=FONT_SIZE,
        color="navy",
        weight="bold",
    )

    color_hatch_map = dict()
    # Fix the legend hatches
    for i, legend_patch in enumerate(ax1.get_legend().get_patches()):
        hatch = hatches[i % len(hatches)]
        legend_patch.set_hatch(f"{hatch}{hatch}")
        color_hatch_map[legend_patch.get_facecolor()] = hatch
        print(f"legend {hatch}")

    def barplot_add_hatches(plot_in_grid, nr_hues, offset=0):
        hatches_used = -1
        bars_hatched = 0
        for bar in plot_in_grid.patches:
            if nr_hues <= 1:
                hatches_used += 1
            else: # with multiple hues, we draw bars with the same hatch in batches
                if bars_hatched % nr_hues == 0:
                    hatches_used += 1
            # if bars_hatched % 7 == 0:
            #     hatches_used += 1
            bars_hatched += 1
            if bar.get_bbox().x0 == 0 and bar.get_bbox().x1 == 0 and bar.get_bbox().y0 == 0 and bar.get_bbox().y1 == 0:
                # skip bars that are not rendered
                continue
            hatch = hatches[(offset + hatches_used) % len(hatches)]
            print(bar, hatches_used, hatch)
            bar.set_hatch(hatch)
            if bars_hatched >= 11:
                break

    # barplot_add_hatches(ax1, 7)
    # patches = 7
    # hues = 2
    # hatched = 0
    # for bar in ax1.patches:
    #     patchNr = (patches * hues) - hatched - 1
    #     hatchNr = patchNr % patches
    #     hatch = hatches[hatchNr % len(hatches)]
    #     bar.set_hatch(hatch)
    #     hatched += 1
    #     breakpoint()
    #     print(f"{patchNr}, {hatchNr}: {hatch}")

    for bar in ax1.patches:
        hatch = color_hatch_map[bar.get_facecolor()]
        bar.set_hatch(hatch)

    ax1.set_xlabel(X1LABEL)
    ax2.set_xlabel(X2LABEL)
    ax1.set_ylabel(YLABEL)

    # plt.ylim(0, 1)
    if not args.logarithmic:
        plt.ylim(bottom=0)
    # ax2.set_xlim(10, 1)
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
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(
        # top=0.9,
        bottom=0.48)
    # fig.tight_layout(rect=(0, 0, 0.3, 1))
    plt.savefig(args.output.name)
    plt.close()

    oob_safety_pct = (avg_verify / (avg_compile + avg_link)) * 100
    reconf_safety_pct = (avg_validate / (avg_validate + avg_jit)) * 100
    print(f"Out of band tasks contain {oob_safety_pct:.1f}% and reconfiguration {reconf_safety_pct:.1f}% of safety related task time.")
    print(f"Total out-of-band time: {total_out_of_band:.1f}ms")
    print(f"Total reconfiguration time: {total_reconfiguration:.1f}ms")




if __name__ == '__main__':
    main()
