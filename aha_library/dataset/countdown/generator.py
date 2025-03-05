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

def random_expression_stack(numbers, leave_out_frac_max=.2):
  """Builds a random valid arithmetic expression using multiple operations, from left only."""
  op_symbols = {operator.add: '+', operator.sub: '-', operator.mul: '*', operator.floordiv: '/'}
  ops, used_ops = list(op_symbols.keys()), set()
  
  random.shuffle(numbers)
  n0 = numbers[0]
  value, expression_arr, used_numbers = n0, [str(n0)], {n0}
  for num in numbers[1:]:
    op = random.choice(ops)
    
    if op == operator.floordiv and (num == 0 or value % num != 0):
      continue # Skip invalid division
    if op == operator.sub and (num > value):
      continue # Skip invalid subtraction
    
    value = op(value, num)
    expression_arr.append(f'{op_symbols[op]}{num})')
    used_ops.add(op)
    used_numbers.add(num)
    
  expression='('*(len(expression_arr)-1) + ''.join(expression_arr)
  if len(used_ops)==0 or len(used_numbers)<len(numbers)*(1-leave_out_frac_max):
    expression=None # Ensure diversity of operations and near full usage (invalidate otherwise)
  return expression, value

def random_expression_tree(numbers, leave_out_frac_max=.2):
  """Builds a random valid arithmetic expression using multiple operations, in a tree (potentially more expressive)"""
  op_symbols = {operator.add: '+', operator.sub: '-', operator.mul: '*', operator.floordiv: '/'}
  ops = list(op_symbols.keys())
  
  random.shuffle(numbers)
  pairs = [ (n, str(n)) for n in numbers ] # Structure for the numbers
  leave_out = int( random.random()*leave_out_frac_max*len(pairs) )
  pairs = pairs[leave_out:]  # Skip the first few, if randomly decided
  
  while len(pairs)>1:
    i = random.randint(0, len(pairs)-2)  # This addresses pair[i] and pair[i+1] - randint(a,b) = [a..b] inclusive
    v0, v1 = pairs[i][0], pairs[i+1][0]
    e0, e1 = pairs[i][1], pairs[i+1][1]
    op = random.choice(ops)
    #print(f"{pairs}@{i} : ({e0}={v0}) {op_symbols[op]} ({v1}={e1})")
    
    if op == operator.floordiv and (v1 == 0 or v0 % v1 != 0):
      continue # Skip invalid division
    if op == operator.sub and (v0 < v1):
      continue # Skip invalid subtraction
    
    pair_new = ( op(v0, v1), f"({e0}{op_symbols[op]}{e1})" )
    pairs.pop(i+1)
    pairs[i]=pair_new
    
  value, expression = pairs[0] # This has the accumulated value, expression
  return expression[1:-1], value  # Remove outermost brackets


def generate_puzzle(seed=None, n_small=4, n_large=2, target_min=100, target_max=999, as_structure=False):
  if seed is not None: 
    random.seed(seed)
  while True: # We'll definitely (!) find one eventually
    numbers = generate_numbers(n_small=n_small, n_large=n_large)
    #expression, target = random_expression_stack(numbers)
    expression, target = random_expression_tree(numbers) # More expressive
    #print(sorted(numbers), target, expression)
    if target_min<=target<=target_max and expression is not None: # result within bounds, and expression_arr valid
      if as_structure:
        return dict(
          numbers=' '.join(str(n) for n in sorted(numbers)),
          target=str(target),
          proof=expression,
        )
      else:        
        return sorted(numbers), target, expression

if __name__ == "__main__":
  numbers, target, expression = generate_puzzle()
  print("Chosen numbers:", sorted(numbers))
  print("Target:", target)
  print("Expression:", expression)
  print("Verification:", eval(expression))
