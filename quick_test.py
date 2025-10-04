#!/usr/bin/env python3
from improved_fact_checker import ImprovedFakeNewsDetector

detector = ImprovedFakeNewsDetector()

tests = [
    'virat kohli is a footballer', 
    'rohit sharma is a kabaddi player',
    'virat kohli has more centuries than sachin tendulkar'
]

for test in tests:
    print(f'\n🧪 Testing: {test}')
    result = detector.comprehensive_analysis(test)
    print(f'✅ Result: {result["final_prediction"]} ({result["final_confidence"]:.0%})')
    print(f'🧠 Reason: {result["reasoning"]}')
    print('=' * 50)