import random
import operator
from itertools import permutations

def generate_numbers(n_small=4, n_large=2):
  """Randomly selects six numbers: a mix of small and large."""
  small_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2 # Each appears twice
  large_numbers = [25, 50, 75, 100]
  
  chosen = random.sample(small_numbers, k=n_small) + random.sample(large_numbers, k=n_large)
  random.shuffle(chosen)

  return chosen

def random_expression(numbers):
  """Builds a random valid arithmetic expression using multiple operations."""
  ops = [operator.add, operator.sub, operator.mul, operator.floordiv]
  op_symbols = {operator.add: '+', operator.sub: '-', operator.mul: '*', operator.floordiv: '/'}
  
  random.shuffle(numbers)

  used_ops = set()
  
  n0 = numbers[0]
  value, expression_arr, used_numbers = n0, [str(n0)], {n0}
  for num in numbers[1:]:
    #available_ops = [op for op in ops 
    #          if op not in used_ops or len(used_ops)<len(ops)]
    #op = random.choice(available_ops)
    op = random.choice(ops)
    
    if op == operator.floordiv and (num == 0 or value % num != 0):
      continue # Skip invalid division
    if op == operator.sub and (num > value):
      continue # Skip invalid subtraction
    
    value = op(value, num)
    expression_arr.append(f'{op_symbols[op]}{num})')
    used_ops.add(op)
    used_numbers.add(num)
    
#    if len(used_ops)==len(ops) and len(used_numbers)>=len(numbers)-1:
#      break # Ensure diversity of operations and near full usage

  if len(used_ops)==0 or len(used_numbers)<len(numbers)*.8:
    expression_arr=[] # Ensure diversity of operations and near full usage (invalidate otherwise)
  
  return expression_arr, value

def generate_puzzle(seed=None, target_min=100, target_max=999, as_structure=False):
  if seed is not None: 
    random.seed(seed)
  while True: # We'll definitely (!) find one eventually
    numbers = generate_numbers()
    expression_arr, target = random_expression(numbers)
    #print(sorted(numbers), target, expression_arr)
    if target_min<=target<=target_max and len(expression_arr)>0: # result within bounds, and expression_arr valid
      expression='('*(len(expression_arr)-1) + ''.join(expression_arr)
      if as_structure:
        return dict(
          question=' '.join(str(n) for n in sorted(numbers)),
          answer=str(target),
          proof=expression
        )
      else:        
        return sorted(numbers), target, expression

if __name__ == "__main__":
  numbers, target, expression = generate_puzzle()
  print("Chosen numbers:", sorted(numbers))
  print("Target:", target)
  print("Expression:", expression)
  print("Verification:", eval(expression))
