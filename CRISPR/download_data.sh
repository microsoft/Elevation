#/bin/bash

# download supplementary tables 18 and 19 from Doench et al. 2016
wget https://images.nature.com/original/nature-assets/nbt/journal/v34/n2/extref/nbt.3437-S2.zip
unzip nbt.3437-S2.zip -x '__MACOSX/*'
mv 'SuppTables/STable 18 CD33_OffTargetdata.xlsx' data/offtarget
mv 'SuppTables/STable 19 FractionActive_dlfc_lookup.xlsx' data/offtarget
rm -rf ./SuppTables

# download supplementary table from Tsai et al. 2015
wget https://images.nature.com/original/nature-assets/nbt/journal/v33/n2/extref/nbt.3117-S2.xlsx
mv nbt.3117-S2.xlsx data/offtarget

# run script to filter CD33 data
python filter_s_table_18.py -d './data/offtarget/STable 18 CD33_OffTargetdata.xlsx' -o './data/offtarget/CD33_data_postfilter.xlsx'

# run script to filter Tsai et al. data
python filter_tsai.py -d './data/offtarget/nbt.3117-S2.xlsx' -o './data/offtarget/Supplementary Table 10.xlsx'
