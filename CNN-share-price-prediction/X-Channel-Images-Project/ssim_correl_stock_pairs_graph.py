import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os, sys
sys.path.append(os.path.abspath('./helper_functions_dir'))
import helper_functions_dir.helper_functions as helper_functions


from parameters import Parameters
from parameters import StockParams

#ssim_encoded_image_results = {'Train:SIVBQ_Eval:SICP': 0.3704536259174347, 'Train:SIVBQ_Eval:ALLY': 0.48212066292762756, 'Train:SIVBQ_Eval:CMA': 0.3736174404621124, 'Train:SIVBQ_Eval:WAL': 0.4726531207561493, 'Train:SIVBQ_Eval:PWBK': 0.05071616545319557, 'Train:SIVBQ_Eval:ZION': 0.38753578066825867, 'Train:SIVBQ_Eval:KEY': 0.4010263979434967, 'Train:SIVBQ_Eval:CUBI': 0.4249544143676758, 'Train:SIVBQ_Eval:OZK': 0.4123328626155853, 'Train:SIVBQ_Eval:CFG': 0.3778620958328247, 'Train:SIVBQ_Eval:RF': 0.3904111087322235, 'Train:SIVBQ_Eval:FITB': 0.4399232864379883, 'Train:SIVBQ_Eval:HBAN': 0.38596341013908386, 'Train:SICP_Eval:ALLY': 0.31234076619148254, 'Train:SICP_Eval:CMA': 0.2613639831542969, 'Train:SICP_Eval:WAL': 0.3275105059146881, 'Train:SICP_Eval:PWBK': 0.09146274626255035, 'Train:SICP_Eval:ZION': 0.2505556643009186, 'Train:SICP_Eval:KEY': 0.24370315670967102, 'Train:SICP_Eval:CUBI': 0.3042617440223694, 'Train:SICP_Eval:OZK': 0.249998539686203, 'Train:SICP_Eval:CFG': 0.21221822500228882, 'Train:SICP_Eval:RF': 0.24619060754776, 'Train:SICP_Eval:FITB': 0.24346484243869781, 'Train:SICP_Eval:HBAN': 0.2658926248550415, 'Train:ALLY_Eval:CMA': 0.4802308678627014, 'Train:ALLY_Eval:WAL': 0.49428844451904297, 'Train:ALLY_Eval:PWBK': 0.06526217609643936, 'Train:ALLY_Eval:ZION': 0.4841848909854889, 'Train:ALLY_Eval:KEY': 0.48306146264076233, 'Train:ALLY_Eval:CUBI': 0.4544219970703125, 'Train:ALLY_Eval:OZK': 0.43080028891563416, 'Train:ALLY_Eval:CFG': 0.48274415731430054, 'Train:ALLY_Eval:RF': 0.42207205295562744, 'Train:ALLY_Eval:FITB': 0.4809723496437073, 'Train:ALLY_Eval:HBAN': 0.4561319947242737, 'Train:CMA_Eval:WAL': 0.533659815788269, 'Train:CMA_Eval:PWBK': 0.07645225524902344, 'Train:CMA_Eval:ZION': 0.7216663360595703, 'Train:CMA_Eval:KEY': 0.6517255306243896, 'Train:CMA_Eval:CUBI': 0.45770058035850525, 'Train:CMA_Eval:OZK': 0.4675182104110718, 'Train:CMA_Eval:CFG': 0.5821740627288818, 'Train:CMA_Eval:RF': 0.5938384532928467, 'Train:CMA_Eval:FITB': 0.6300323009490967, 'Train:CMA_Eval:HBAN': 0.5203545093536377, 'Train:WAL_Eval:PWBK': 0.05302180349826813, 'Train:WAL_Eval:ZION': 0.5266730189323425, 'Train:WAL_Eval:KEY': 0.5666338801383972, 'Train:WAL_Eval:CUBI': 0.504608154296875, 'Train:WAL_Eval:OZK': 0.5270249247550964, 'Train:WAL_Eval:CFG': 0.5098334550857544, 'Train:WAL_Eval:RF': 0.5368549823760986, 'Train:WAL_Eval:FITB': 0.5580867528915405, 'Train:WAL_Eval:HBAN': 0.5140599012374878, 'Train:PWBK_Eval:ZION': 0.07537403702735901, 'Train:PWBK_Eval:KEY': 0.06922992318868637, 'Train:PWBK_Eval:CUBI': 0.03235507756471634, 'Train:PWBK_Eval:OZK': 0.04861287400126457, 'Train:PWBK_Eval:CFG': 0.056576650589704514, 'Train:PWBK_Eval:RF': 0.06506425887346268, 'Train:PWBK_Eval:FITB': 0.05417218804359436, 'Train:PWBK_Eval:HBAN': 0.051518842577934265, 'Train:ZION_Eval:KEY': 0.6150786876678467, 'Train:ZION_Eval:CUBI': 0.46293795108795166, 'Train:ZION_Eval:OZK': 0.453029990196228, 'Train:ZION_Eval:CFG': 0.5414749383926392, 'Train:ZION_Eval:RF': 0.6064474582672119, 'Train:ZION_Eval:FITB': 0.5648069977760315, 'Train:ZION_Eval:HBAN': 0.4894666075706482, 'Train:KEY_Eval:CUBI': 0.5418223738670349, 'Train:KEY_Eval:OZK': 0.5948302149772644, 'Train:KEY_Eval:CFG': 0.7987590432167053, 'Train:KEY_Eval:RF': 0.6577513813972473, 'Train:KEY_Eval:FITB': 0.7285614609718323, 'Train:KEY_Eval:HBAN': 0.6704742312431335, 'Train:CUBI_Eval:OZK': 0.5304791331291199, 'Train:CUBI_Eval:CFG': 0.5309602618217468, 'Train:CUBI_Eval:RF': 0.471529483795166, 'Train:CUBI_Eval:FITB': 0.5471972227096558, 'Train:CUBI_Eval:HBAN': 0.5295976400375366, 'Train:OZK_Eval:CFG': 0.5620689392089844, 'Train:OZK_Eval:RF': 0.6122348308563232, 'Train:OZK_Eval:FITB': 0.6004860997200012, 'Train:OZK_Eval:HBAN': 0.6062372922897339, 'Train:CFG_Eval:RF': 0.6275138854980469, 'Train:CFG_Eval:FITB': 0.7221697568893433, 'Train:CFG_Eval:HBAN': 0.6877825260162354, 'Train:RF_Eval:FITB': 0.6540189981460571, 'Train:RF_Eval:HBAN': 0.6430084705352783, 'Train:FITB_Eval:HBAN': 0.673761785030365} 
mse_encoded_image_results = {'Train:SIVBQ_Eval:SICP': 0.967961311340332, 'Train:SIVBQ_Eval:ALLY': 0.6341077089309692, 'Train:SIVBQ_Eval:CMA': 0.966698944568634, 'Train:SIVBQ_Eval:WAL': 0.919901430606842, 'Train:SIVBQ_Eval:PWBK': 2.1913180351257324, 'Train:SIVBQ_Eval:ZION': 0.8933643698692322, 'Train:SIVBQ_Eval:KEY': 1.0193514823913574, 'Train:SIVBQ_Eval:CUBI': 0.8882917761802673, 'Train:SIVBQ_Eval:OZK': 1.0229897499084473, 'Train:SIVBQ_Eval:CFG': 1.0999642610549927, 'Train:SIVBQ_Eval:RF': 1.0201752185821533, 'Train:SIVBQ_Eval:FITB': 0.8816253542900085, 'Train:SIVBQ_Eval:HBAN': 1.0916861295700073, 'Train:SICP_Eval:ALLY': 0.9656223654747009, 'Train:SICP_Eval:CMA': 1.0638872385025024, 'Train:SICP_Eval:WAL': 0.9635366797447205, 'Train:SICP_Eval:PWBK': 2.072747230529785, 'Train:SICP_Eval:ZION': 1.043237328529358, 'Train:SICP_Eval:KEY': 1.2217267751693726, 'Train:SICP_Eval:CUBI': 0.8888277411460876, 'Train:SICP_Eval:OZK': 1.1695530414581299, 'Train:SICP_Eval:CFG': 1.280379295349121, 'Train:SICP_Eval:RF': 1.0939781665802002, 'Train:SICP_Eval:FITB': 1.079626441001892, 'Train:SICP_Eval:HBAN': 1.0683106184005737, 'Train:ALLY_Eval:CMA': 0.6189922094345093, 'Train:ALLY_Eval:WAL': 0.6617571115493774, 'Train:ALLY_Eval:PWBK': 1.7653096914291382, 'Train:ALLY_Eval:ZION': 0.6183688044548035, 'Train:ALLY_Eval:KEY': 0.6681540608406067, 'Train:ALLY_Eval:CUBI': 0.7169913053512573, 'Train:ALLY_Eval:OZK': 0.734201192855835, 'Train:ALLY_Eval:CFG': 0.6845976710319519, 'Train:ALLY_Eval:RF': 0.7248267531394958, 'Train:ALLY_Eval:FITB': 0.5808438658714294, 'Train:ALLY_Eval:HBAN': 0.7100423574447632, 'Train:CMA_Eval:WAL': 0.566311240196228, 'Train:CMA_Eval:PWBK': 1.8569583892822266, 'Train:CMA_Eval:ZION': 0.2288840413093567, 'Train:CMA_Eval:KEY': 0.37251877784729004, 'Train:CMA_Eval:CUBI': 0.7304683923721313, 'Train:CMA_Eval:OZK': 0.5544276237487793, 'Train:CMA_Eval:CFG': 0.4934699237346649, 'Train:CMA_Eval:RF': 0.37374985218048096, 'Train:CMA_Eval:FITB': 0.3323591649532318, 'Train:CMA_Eval:HBAN': 0.5494558811187744, 'Train:WAL_Eval:PWBK': 1.8670406341552734, 'Train:WAL_Eval:ZION': 0.6160516738891602, 'Train:WAL_Eval:KEY': 0.5044924020767212, 'Train:WAL_Eval:CUBI': 0.5080517530441284, 'Train:WAL_Eval:OZK': 0.5026537775993347, 'Train:WAL_Eval:CFG': 0.5771768093109131, 'Train:WAL_Eval:RF': 0.47701752185821533, 'Train:WAL_Eval:FITB': 0.45534956455230713, 'Train:WAL_Eval:HBAN': 0.4959755539894104, 'Train:PWBK_Eval:ZION': 1.8955775499343872, 'Train:PWBK_Eval:KEY': 1.8722209930419922, 'Train:PWBK_Eval:CUBI': 1.9717082977294922, 'Train:PWBK_Eval:OZK': 1.9702930450439453, 'Train:PWBK_Eval:CFG': 1.9226330518722534, 'Train:PWBK_Eval:RF': 1.8552507162094116, 'Train:PWBK_Eval:FITB': 1.9034991264343262, 'Train:PWBK_Eval:HBAN': 1.9540091753005981, 'Train:ZION_Eval:KEY': 0.3939094543457031, 'Train:ZION_Eval:CUBI': 0.7768716216087341, 'Train:ZION_Eval:OZK': 0.5761479139328003, 'Train:ZION_Eval:CFG': 0.5039316415786743, 'Train:ZION_Eval:RF': 0.3727925717830658, 'Train:ZION_Eval:FITB': 0.3884313404560089, 'Train:ZION_Eval:HBAN': 0.6059086918830872, 'Train:KEY_Eval:CUBI': 0.5447567701339722, 'Train:KEY_Eval:OZK': 0.4189700782299042, 'Train:KEY_Eval:CFG': 0.14734025299549103, 'Train:KEY_Eval:RF': 0.32513996958732605, 'Train:KEY_Eval:FITB': 0.2524164915084839, 'Train:KEY_Eval:HBAN': 0.33558332920074463, 'Train:CUBI_Eval:OZK': 0.5461110472679138, 'Train:CUBI_Eval:CFG': 0.5189056396484375, 'Train:CUBI_Eval:RF': 0.6237321496009827, 'Train:CUBI_Eval:FITB': 0.481838583946228, 'Train:CUBI_Eval:HBAN': 0.4607200026512146, 'Train:OZK_Eval:CFG': 0.4392397403717041, 'Train:OZK_Eval:RF': 0.3811459243297577, 'Train:OZK_Eval:FITB': 0.36907264590263367, 'Train:OZK_Eval:HBAN': 0.36242765188217163, 'Train:CFG_Eval:RF': 0.37570250034332275, 'Train:CFG_Eval:FITB': 0.28135183453559875, 'Train:CFG_Eval:HBAN': 0.30566489696502686, 'Train:RF_Eval:FITB': 0.3148455023765564, 'Train:RF_Eval:HBAN': 0.33215299248695374, 'Train:FITB_Eval:HBAN': 0.32528114318847656}

def plot_metric_stock_pairs(dtw_dist_dict,experiment_name, run_id):
    print("dtw dict keys",dtw_dist_dict.keys())
    sorted_mse_results = {k: v for k, v in sorted(mse_encoded_image_results.items(), key=lambda item: item[1], reverse=True)}
    
    keys = list(sorted_mse_results.keys())
    print("Keys",keys)
    mse_values = list(sorted_mse_results.values())
    
    dtw_dist_dict_values = [dtw_dist_dict[key] for key in keys]

    fig, ax1 = plt.subplots(figsize=(14, 10))  # Increase figure size

    # Plot SSIM as bars
    bars = ax1.bar(keys, mse_values, color='lightblue', label='SSIM')

    # Add values above each bar and rotate them vertically
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{height:.2f}', 
                 ha='center', va='bottom', fontsize=9, rotation=90, color='black')  
    
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x:.2f}'))
    ax1.set_ylabel('Image MSE Score')
    
    ax1.set_xticks(range(len(keys)))
    ax1.set_xticklabels(keys, rotation=90, ha='center', fontsize=8)
    
    ax1.invert_xaxis()

    plt.margins(x=0.05, y=0.1)

    plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.25) 
    
    ax2 = ax1.twinx()
    #ax2.plot(keys, pearson_values, color='red', label='Pearson Correlation', marker='o', linestyle='-', linewidth=2)
    ax2.plot(keys, dtw_dist_dict_values, color='red', label='DTW Distances', marker='o', linestyle='-', linewidth=2)
    
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x:.4f}'))
    #ax2.set_ylabel('Pearson Correlation')
    ax2.set_ylabel('DTW Distances')

    #legends
    #legend log price series correl vs images ssim scores
    # bars_legend = ax1.bar(keys, ssim_values, color='lightblue', label='Images SSIM Scores')
    # line_legend, = ax2.plot(keys, pearson_values, color='red', label='Log Price Spearman Correlation')
    # fig.legend([bars_legend, line_legend], ['Images SSIM Scores', 'Log Price Spearman Correlation'], loc="upper left")

    #legend log price DTW distances vs images ssim scores
    # bars_legend = ax1.bar(keys, mse_values, color='lightblue', label='Images SSIM Scores')
    # line_legend, = ax2.plot(keys, dtw_dist_dict_values, color='red', label='Log Price Series DTW Distances')
    # fig.legend([bars_legend, line_legend], ['Images SSIM Scores', 'Log Price Series DTW Distances'], loc="upper left")

    bars_legend = ax1.bar(keys, mse_values, color='lightblue', label='Images MSE')
    line_legend, = ax2.plot(keys, dtw_dist_dict_values, color='red', label='Log Price Series DTW Distances')
    fig.legend([bars_legend, line_legend], ['Images MSE', 'Log Price Series DTW Distances'], loc="upper left")

    # Set labels and title
    plt.title(f'Images MSE and Log Price Series DTW Distances')

    # Tight layout to adjust for all elements
    plt.tight_layout()
    print("Saving now")
    plt.savefig('mse_dtw_distance_pairs.png', dpi=300)
    plt.show()

# def plot_metric_stock_pairs(pearson_img_correl_dict,experiment_name, run_id):
#     print("****pearson",pearson_img_correl_dict)
#     sorted_dtw_results = {k: v for k, v in sorted(ssim_encoded_image_results.items(), key=lambda item: item[1], reverse=True)}
    
#     #print(sorted_dtw_results)
    
#     keys = list(sorted_dtw_results.keys())
#     values = list(sorted_dtw_results.values())

#     fig, ax = plt.subplots(figsize=(14, 10))  # Increase figure size

#     bars = plt.bar(keys, values, color='lightblue')  # Create vertical bar chart

#     # Add values above each bar and rotate them vertically
#     for bar in bars:
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width() / 2, height + 1000, f'{int(height):,}', 
#                 ha='center', va='bottom', fontsize=9, rotation=90, color='black')  # Rotated values vertically
    
#     # Format y-axis values with commas as thousands separators
#     #ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
#     ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x:.2f}'))

#     # Rotate x-tick labels vertically for better fit
#     plt.xticks(rotation=90, ha='center', fontsize=8)

#     # Invert the x-axis to flip the order of bars
#     ax.invert_xaxis()

#     # Set margins to give more space around the bars
#     plt.margins(x=0.05, y=0.1)  # Adjust x and y margins (5% for x and 10% for y)

#     # Use subplots_adjust to give more room around the figure
#     plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)  # Customize padding around the plot

#     # Set labels and title
#     plt.xlabel('Train vs Evaluate Pairs')
#     plt.ylabel('SSIM Score')
#     plt.title(f'Structural Similarity Index Measure for Encoded Images')

#     plt.tight_layout()  # Adjust layout to ensure everything fits
#     plt.savefig('sorted_ssim_pairs.png', dpi=300)

#     plt.show()

    # helper_functions.write_and_log_plt(fig, None,
    #     f"DTW_Distance_Train_Eval_Pairs",
    #     f"DTW_Distance_Train_Eval_Pairs", experiment_name, run_id)
    
#plot_metric_stock_pairs(ssim_encoded_image_results, None,None)



# dtw_encoded_image_results = {
#     'Train:SIVBQ_Eval:SICP:': 688940.8,
#     'Train:SIVBQ_Eval:ALLY:': 554159.4,
#     'Train:SIVBQ_Eval:CMA:': 632659.6,
#     'Train:SIVBQ_Eval:WAL:': 622864.5,
#     'Train:SIVBQ_Eval:PWBK:': 790642.5,
#     'Train:SIVBQ_Eval:ZION:': 590456.0,
#     'Train:SIVBQ_Eval:KEY:': 578979.0,
#     'Train:SIVBQ_Eval:CUBI:': 620348.4,
#     'Train:SIVBQ_Eval:OZK:': 591428.1,
#     'Train:SIVBQ_Eval:CFG:': 518205.5,
#     'Train:SIVBQ_Eval:RF:': 586858.3,
#     'Train:SIVBQ_Eval:FITB:': 543201.3,
#     'Train:SIVBQ_Eval:HBAN:': 540474.5,
#     'Train:SICP_Eval:ALLY:': 607786.6,
#     'Train:SICP_Eval:CMA:': 515969.2,
#     'Train:SICP_Eval:WAL:': 622428.2,
#     'Train:SICP_Eval:PWBK:': 696929.0,
#     'Train:SICP_Eval:ZION:': 595070.6,
#     'Train:SICP_Eval:KEY:': 560828.9,
#     'Train:SICP_Eval:CUBI:': 583422.7,
#     'Train:SICP_Eval:OZK:': 530594.9,
#     'Train:SICP_Eval:CFG:': 529967.6,
#     'Train:SICP_Eval:RF:': 617889.4,
#     'Train:SICP_Eval:FITB:': 631344.8,
#     'Train:SICP_Eval:HBAN:': 575080.3,
#     'Train:ALLY_Eval:CMA:': 593527.8,
#     'Train:ALLY_Eval:WAL:': 580515.6,
#     'Train:ALLY_Eval:PWBK:': 786969.0,
#     'Train:ALLY_Eval:ZION:': 562057.8,
#     'Train:ALLY_Eval:KEY:': 658975.8,
#     'Train:ALLY_Eval:CUBI:': 572030.6,
#     'Train:ALLY_Eval:OZK:': 578515.0,
#     'Train:ALLY_Eval:CFG:': 644055.4,
#     'Train:ALLY_Eval:RF:': 574249.1,
#     'Train:ALLY_Eval:FITB:': 588426.2,
#     'Train:ALLY_Eval:HBAN:': 617990.3,
#     'Train:CMA_Eval:WAL:': 590181.7,
#     'Train:CMA_Eval:PWBK:': 823470.8,
#     'Train:CMA_Eval:ZION:': 550747.5,
#     'Train:CMA_Eval:KEY:': 430258.8,
#     'Train:CMA_Eval:CUBI:': 572986.3,
#     'Train:CMA_Eval:OZK:': 538961.5,
#     'Train:CMA_Eval:CFG:': 502847.1,
#     'Train:CMA_Eval:RF:': 476708.2,
#     'Train:CMA_Eval:FITB:': 509987.0,
#     'Train:CMA_Eval:HBAN:': 673414.8,
#     'Train:WAL_Eval:PWBK:': 748966.6,
#     'Train:WAL_Eval:ZION:': 499388.4,
#     'Train:WAL_Eval:KEY:': 450008.6,
#     'Train:WAL_Eval:CUBI:': 605059.4,
#     'Train:WAL_Eval:OZK:': 634222.2,
#     'Train:WAL_Eval:CFG:': 477928.8,
#     'Train:WAL_Eval:RF:': 479693.9,
#     'Train:WAL_Eval:FITB:': 419561.7,
#     'Train:WAL_Eval:HBAN:': 543295.7,
#     'Train:PWBK_Eval:ZION:': 807358.9,
#     'Train:PWBK_Eval:KEY:': 734260.9,
#     'Train:PWBK_Eval:CUBI:': 819935.6,
#     'Train:PWBK_Eval:OZK:': 826825.8,
#     'Train:PWBK_Eval:CFG:': 755732.9,
#     'Train:PWBK_Eval:RF:': 776267.8,
#     'Train:PWBK_Eval:FITB:': 765456.7,
#     'Train:PWBK_Eval:HBAN:': 745467.4,
#     'Train:ZION_Eval:KEY:': 487253.3,
#     'Train:ZION_Eval:CUBI:': 558999.9,
#     'Train:ZION_Eval:OZK:': 568247.7,
#     'Train:ZION_Eval:CFG:': 496612.2,
#     'Train:ZION_Eval:RF:': 464933.4,
#     'Train:ZION_Eval:FITB:': 513518.5,
#     'Train:ZION_Eval:HBAN:': 554048.9,
#     'Train:KEY_Eval:CUBI:': 581914.4,
#     'Train:KEY_Eval:OZK:': 513475.5,
#     'Train:KEY_Eval:CFG:': 450944.1,
#     'Train:KEY_Eval:RF:': 412312.6,
#     'Train:KEY_Eval:FITB:': 498083.8,
#     'Train:KEY_Eval:HBAN:': 538184.4,
#     'Train:CUBI_Eval:OZK:': 679741.3,
#     'Train:CUBI_Eval:CFG:': 454641.2,
#     'Train:CUBI_Eval:RF:': 571838.4,
#     'Train:CUBI_Eval:FITB:': 715256.7,
#     'Train:CUBI_Eval:HBAN:': 517124.7,
#     'Train:OZK_Eval:CFG:': 506214.3,
#     'Train:OZK_Eval:RF:': 524147.5,
#     'Train:OZK_Eval:FITB:': 596603.0,
#     'Train:OZK_Eval:HBAN:': 549125.0,
#     'Train:CFG_Eval:RF:': 490040.1,
#     'Train:CFG_Eval:FITB:': 334271.0,
#     'Train:CFG_Eval:HBAN:': 521759.8,
#     'Train:RF_Eval:FITB:': 380215.6,
#     'Train:RF_Eval:HBAN:': 519956.4,
#     'Train:FITB_Eval:HBAN:': 527207.0
# }
