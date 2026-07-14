import json

vi_path = r"c:\Users\ADMIN\OneDrive\Máy tính\mob_sleep\assets\translations\vi.json"
with open(vi_path, 'r', encoding='utf-8') as f:
    d = json.load(f)

# Pichot Fatigue questions
pichot_keys = [
    'i_feel_low_on_energy',
    'everything_requires_effort_from_me',
    'i_feel_weak_in_some_parts_of_my_body',
    'i_feel_heavy_in_my_legs_or_arms',
    'i_feel_tired_without_any_explanation',
    'i_want_to_lie_down',
    'i_find_it_difficult_to_concentrate',
    'i_feel_tired_heavy_sluggish'
]

# Pichot QD questions
pichot_qd_keys = [
    'i_find_it_hard_to_escape_negative_thoughts_in_my_head',
    'i_feel_drained_of_energy',
    'i_no_longer_enjoy_things_i_used_to_love_or_care_about',
    'i_feel_disappointed_and_down_on_myself',
    'i_feel_inhibited_and_obstructed_when_trying_to_do_something',
    'i_currently_feel_less_happy_than_others',
    'deeply_bored_or_uninterested',
    'i_feel_like_i_have_to_force_myself_to_do_anything',
    'my_mental_clarity_is_worse_than_before',
    'i_currently_feel_very_sad',
    'i_lack_the_ability_to_make_decisions_and_regulate_myself',
    'i_struggle_with_tasks_i_used_to_do_easily',
    'life_feels_empty_to_me_now'
]

with open('questionnaire_check.txt', 'w', encoding='utf-8') as out:
    out.write("=== PICHOT FATIGUE (8 questions) ===\n")
    for i, k in enumerate(pichot_keys):
        out.write(f"  Q{i+1}: {d.get(k, 'NOT FOUND')}\n")
    
    out.write("\n=== PICHOT QD (13 questions) ===\n")
    for i, k in enumerate(pichot_qd_keys):
        out.write(f"  Q{i+1}: {d.get(k, 'NOT FOUND')}\n")

print("SUCCESS")
