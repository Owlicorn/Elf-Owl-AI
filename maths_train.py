# math_data_generator.py
import random
import json
import argparse
from typing import List, Dict, Tuple

class MathDataGenerator:
    def __init__(self):
        self.operators = ['+', '-', '*', '/']
        
    def generate_arithmetic_expression(self, max_numbers: int = 3, max_value: int = 50) -> Tuple[str, str, str]:
        """Generate arithmetic expression with solution and thinking steps"""
        num_numbers = random.randint(2, max_numbers)
        numbers = [random.randint(1, max_value) for _ in range(num_numbers)]
        operators = [random.choice(self.operators) for _ in range(num_numbers - 1)]
        
        # Build expression
        expression = str(numbers[0])
        for i, op in enumerate(operators):
            expression += f" {op} {numbers[i + 1]}"
        
        # Calculate result with step-by-step thinking
        thinking_steps = []
        current_value = numbers[0]
        
        for i in range(len(operators)):
            next_num = numbers[i + 1]
            op = operators[i]
            
            if op == '+':
                result = current_value + next_num
                thinking_steps.append(f"Add {current_value} + {next_num} = {result}")
                current_value = result
            elif op == '-':
                result = current_value - next_num
                thinking_steps.append(f"Subtract {current_value} - {next_num} = {result}")
                current_value = result
            elif op == '*':
                result = current_value * next_num
                thinking_steps.append(f"Multiply {current_value} × {next_num} = {result}")
                current_value = result
            elif op == '/':
                if next_num != 0:
                    result = current_value / next_num
                    thinking_steps.append(f"Divide {current_value} ÷ {next_num} = {result}")
                    current_value = result
                else:
                    thinking_steps.append(f"Cannot divide by zero, skip division")
        
        thinking = " | ".join(thinking_steps)
        return f"Calculate: {expression}", thinking, str(current_value)
    
    def generate_algebra_equation(self) -> Tuple[str, str, str]:
        """Generate algebraic equation with solution and thinking"""
        # Simple linear equation: ax + b = c
        a = random.randint(1, 10)
        b = random.randint(1, 20)
        c = random.randint(b + 1, 50)
        
        equation = f"{a}x + {b} = {c}"
        
        thinking_steps = [
            f"Equation: {a}x + {b} = {c}",
            f"Subtract {b} from both sides: {a}x = {c - b}",
            f"Divide both sides by {a}: x = {c - b} / {a}",
            f"Solution: x = {(c - b) / a}"
        ]
        
        thinking = " | ".join(thinking_steps)
        solution = str((c - b) / a)
        return f"Solve for x: {equation}", thinking, solution
    
    def generate_fraction_problem(self) -> Tuple[str, str, str]:
        """Generate fraction problems with thinking"""
        num1 = random.randint(1, 10)
        den1 = random.randint(2, 10)
        num2 = random.randint(1, 10)
        den2 = random.randint(2, 10)
        
        operation = random.choice(['+', '-', '*', '/'])
        
        if operation == '+':
            expression = f"{num1}/{den1} + {num2}/{den2}"
            result_num = num1 * den2 + num2 * den1
            result_den = den1 * den2
            thinking = f"Find common denominator: {den1}×{den2} = {result_den} | Adjust numerators: {num1}×{den2} = {num1*den2}, {num2}×{den1} = {num2*den1} | Add: {num1*den2} + {num2*den1} = {result_num} | Result: {result_num}/{result_den}"
            
        elif operation == '-':
            expression = f"{num1}/{den1} - {num2}/{den2}"
            result_num = num1 * den2 - num2 * den1
            result_den = den1 * den2
            thinking = f"Find common denominator: {den1}×{den2} = {result_den} | Adjust numerators: {num1}×{den2} = {num1*den2}, {num2}×{den1} = {num2*den1} | Subtract: {num1*den2} - {num2*den1} = {result_num} | Result: {result_num}/{result_den}"
            
        elif operation == '*':
            expression = f"{num1}/{den1} × {num2}/{den2}"
            result_num = num1 * num2
            result_den = den1 * den2
            thinking = f"Multiply numerators: {num1} × {num2} = {result_num} | Multiply denominators: {den1} × {den2} = {result_den} | Result: {result_num}/{result_den}"
            
        else:  # division
            expression = f"{num1}/{den1} ÷ {num2}/{den2}"
            result_num = num1 * den2
            result_den = den1 * num2
            thinking = f"Multiply by reciprocal: {num1}/{den1} × {den2}/{num2} | Multiply numerators: {num1} × {den2} = {result_num} | Multiply denominators: {den1} × {num2} = {result_den} | Result: {result_num}/{result_den}"
        
        # Simplify
        gcd_val = self._gcd(result_num, result_den)
        simplified_num = result_num // gcd_val
        simplified_den = result_den // gcd_val
        
        if simplified_den == 1:
            solution = str(simplified_num)
            thinking += f" | Simplify: {result_num}/{result_den} = {simplified_num}"
        else:
            solution = f"{simplified_num}/{simplified_den}"
            thinking += f" | Simplify: {result_num}/{result_den} = {simplified_num}/{simplified_den}"
        
        return f"Calculate: {expression}", thinking, solution
    
    def _gcd(self, a: int, b: int) -> int:
        """Calculate greatest common divisor"""
        while b:
            a, b = b, a % b
        return a
    
    def generate_geometry_problem(self) -> Tuple[str, str, str]:
        """Generate geometry problems with thinking"""
        shapes = ['rectangle', 'triangle', 'circle']
        shape = random.choice(shapes)
        
        if shape == 'rectangle':
            length = random.randint(5, 20)
            width = random.randint(3, 15)
            input_text = f"Find area of rectangle with length {length} and width {width}"
            solution = length * width
            thinking = f"Area of rectangle = length × width = {length} × {width} = {solution}"
            
        elif shape == 'triangle':
            base = random.randint(6, 15)
            height = random.randint(4, 12)
            input_text = f"Find area of triangle with base {base} and height {height}"
            solution = 0.5 * base * height
            thinking = f"Area of triangle = ½ × base × height = 0.5 × {base} × {height} = {solution}"
            
        else:  # circle
            radius = random.randint(3, 10)
            input_text = f"Find area of circle with radius {radius}"
            solution = 3.14159 * radius * radius
            thinking = f"Area of circle = π × radius² = 3.14159 × {radius}² = {solution:.2f}"
        
        return input_text, thinking, str(round(solution, 2))
    
    def generate_percentage_problem(self) -> Tuple[str, str, str]:
        """Generate percentage problems with thinking"""
        problem_type = random.choice(['find_percentage', 'find_original', 'find_percentage_change'])
        
        if problem_type == 'find_percentage':
            number = random.randint(10, 200)
            percentage = random.randint(5, 95)
            input_text = f"Find {percentage}% of {number}"
            solution = number * percentage / 100
            thinking = f"Calculate {percentage}% of {number} = ({percentage}/100) × {number} = {percentage/100} × {number} = {solution}"
            
        elif problem_type == 'find_original':
            percentage = random.randint(10, 90)
            part = random.randint(20, 100)
            original = part / (percentage / 100)
            input_text = f"If {percentage}% of a number is {part}, what is the number?"
            thinking = f"Let the number be x | {percentage}% of x = {part} | ({percentage}/100) × x = {part} | x = {part} ÷ ({percentage}/100) = {part} × (100/{percentage}) = {original}"
            solution = str(round(original, 2))
            
        else:  # percentage change
            original = random.randint(50, 200)
            change = random.randint(10, 80)
            increase = random.choice([True, False])
            
            if increase:
                new_value = original * (1 + change/100)
                input_text = f"If a number increases from {original} by {change}%, what is the new value?"
                thinking = f"Increase = {change}% of {original} = {change/100} × {original} = {original * change/100} | New value = {original} + {original * change/100} = {new_value}"
            else:
                new_value = original * (1 - change/100)
                input_text = f"If a number decreases from {original} by {change}%, what is the new value?"
                thinking = f"Decrease = {change}% of {original} = {change/100} × {original} = {original * change/100} | New value = {original} - {original * change/100} = {new_value}"
            
            solution = str(round(new_value, 2))
        
        return input_text, thinking, solution
    
    def generate_exponent_problem(self) -> Tuple[str, str, str]:
        """Generate exponent problems with thinking"""
        base = random.randint(2, 10)
        exponent = random.randint(2, 5)
        
        input_text = f"Calculate {base}^{exponent}"
        solution = base ** exponent
        
        # Show step-by-step multiplication
        steps = []
        current = 1
        for i in range(exponent):
            current *= base
            steps.append(f"{base}" + (f" × {base}" * i) + f" = {current}")
        
        thinking = " | ".join(steps)
        return input_text, thinking, str(solution)
    
    def generate_training_example(self) -> Dict[str, str]:
        """Generate a complete training example with only input, thinking, output"""
        problem_types = [
            self.generate_arithmetic_expression,
            self.generate_algebra_equation,
            self.generate_fraction_problem,
            self.generate_geometry_problem,
            self.generate_percentage_problem,
            self.generate_exponent_problem
        ]
        
        generator_func = random.choice(problem_types)
        input_text, thinking, output = generator_func()
        
        return {
            "input": input_text,
            "thinking": thinking,
            "output": output
        }
    
    def generate_dataset(self, num_examples: int = 1000, output_file: str = "math_training_data.json"):
        """Generate a complete dataset of math problems"""
        print(f"🔢 Generating {num_examples} math training examples...")
        
        dataset = []
        
        for i in range(num_examples):
            example = self.generate_training_example()
            dataset.append(example)
            
            if (i + 1) % 100 == 0:
                print(f"✅ Generated {i + 1}/{num_examples} examples")
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(dataset)} math training examples to {output_file}")
        
        # Print sample
        print("\n📝 Sample generated data:")
        print(json.dumps(dataset[0], indent=2))
        
        return dataset

def main():
    parser = argparse.ArgumentParser(description='Generate math training data')
    parser.add_argument('--examples', type=int, default=1000, 
                       help='Number of examples to generate (default: 1000)')
    parser.add_argument('--output', type=str, default='math_training_data.json',
                       help='Output file name (default: math_training_data.json)')
    
    args = parser.parse_args()
    
    generator = MathDataGenerator()
    generator.generate_dataset(
        num_examples=args.examples,
        output_file=args.output
    )

if __name__ == "__main__":
    main()