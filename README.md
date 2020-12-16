# test_code
The rep is for testing code.

## usage

1. `git clone https://github.com/calibertytz/test_code.git` 

2. `pip install git+https://github.com/rwbfd/OpenCompetitionV2.git@master`



use google cloud to create 3 VM instances to run respectly.

 `python data_prepare.py `

to get data prepared firstly.

then for every VM instance,

`nohup python test_file.py 'dart' > out.log 2>&1 &`

`nohup python test_file.py 'goss' > out.log 2>&1 &`

`nohup python test_file.py 'gbdt' > out.log 2>&1 &`