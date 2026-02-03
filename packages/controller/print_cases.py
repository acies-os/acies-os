import itertools

_geo = ['1', '0']
_mic = ['1', '0']
_infer = ['2', 'g', 'm', '0']
print(f'# {"GEO":^3} {"MIC":^3} {"INF":^3}')
for x in itertools.product(_geo, _mic, _infer):
    print(f'# {x[0]:^3} {x[1]:^3} {x[2]:^3}  =>')
