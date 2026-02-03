import json
import re

xx = {
    'service_types': {
        'geo': '(?:.*[^_]+|^)(geo).*',
        'mic': '(?:.*[^_]+|^)(mic).*',
        'infer': '.*((?:vfm|mae|ds)(?:_geo|_mic)*).*',
    },
    'ok_states': [
        ('mic', 'geo', 'vfm'),
        ('mic', 'geo', 'mae'),
        ('mic', 'geo', 'ds'),
        ('mic', 'vfm_mic'),
        ('mic', 'mae_mic'),
        ('mic', 'ds_mic'),
        ('geo', 'vfm_geo'),
        ('geo', 'mae_geo'),
        ('geo', 'ds_geo'),
    ],
    'service_failures': [
        ('mic', 'geo', 'vfm_geo'),
        ('mic', 'geo', 'mae_geo'),
        ('mic', 'geo', 'ds_geo'),
        ('mic', 'geo', 'vfm_mic'),
        ('mic', 'geo', 'mae_mic'),
        ('mic', 'geo', 'ds_mic'),
        ('mic', 'geo'),
        ('mic', 'vfm'),
        ('mic', 'mae'),
        ('mic', 'ds'),
        ('mic', 'vfm_geo'),
        ('mic', 'mae_geo'),
        ('mic', 'ds_geo'),
        ('mic',),
        ('geo', 'vfm'),
        ('geo', 'mae'),
        ('geo', 'ds'),
        ('geo', 'vfm_mic'),
        ('geo', 'mae_mic'),
        ('geo', 'ds_mic'),
        ('geo',),
    ],
    'service_failover': 'infer',
    'node_failures': [
        (),
        ('vfm',),
        ('mae',),
        ('ds',),
        ('vfm_geo',),
        ('mae_geo',),
        ('ds_geo',),
        ('vfm_mic',),
        ('mae_mic',),
        ('ds_mic',),
    ],
    'node_failover': 'activate_backup_node',
}

with open('execution_plan.json', 'w') as f:
    json.dump(xx, f)

with open('execution_plan.json') as f:
    plan = json.load(f)

topics = [
    'rs1/geo',
    'rs1/mic',
    'rs1/vfm',
    'rs1/vfm_geo',
    'rs1/vfm_mic',
    'rs1/mae',
    'rs1/mae_geo',
    'rs1/mae_mic',
    'rs1/ds',
    'rs1/ds_geo',
    'rs1/ds_mic',
    'rs1/backup/rs10/vfm_geo',
    'cp/controller',
    'cp/ash',
    'geo',
    'mic',
    'vfm',
    'vfm_geo',
    'vfm_mic',
    'mae',
    'mae_geo',
    'mae_mic',
    'ds',
    'ds_geo',
    'ds_mic',
    'backup/rs10/vfm_geo',
    'controller',
    'ash',
]

# print(plan)

for x in topics:
    print(x, ':', end=' ')
    for pp_name, pp in plan['service_types'].items():
        m = re.match(pp, x)
        try:
            print(pp_name, m.group(1))
            break
        except AttributeError:
            continue
    else:
        print()

print('---------------------')
with open('execution_plan.json') as f:
    self_plan = json.load(f)
    self_plan['ok_states'] = [tuple(sorted(x)) for x in self_plan['ok_states']]
print(self_plan)
