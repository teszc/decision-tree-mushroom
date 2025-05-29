import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_step_safely(step_name, step_function):
    """
    Safely run a step with proper error handling.
    """
    print(f"\n Running: {step_name}")
    print("-" * 50)
    
    try:
        step_function()
        print(f"Completed: {step_name}")
        return True
    except ImportError as e:
        print(f"Import Error in {step_name}: {str(e)}")
        print("This usually means missing dependencies or file structure issues")
        return False
    except FileNotFoundError as e:
        print(f"File Not Found in {step_name}: {str(e)}")
        print("Make sure data files are in the correct location")
        return False
    except Exception as e:
        print(f"Error in {step_name}: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return False

def check_dependencies():
    print("Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"{package} is available")
        except ImportError:
            missing_packages.append(package)
            print(f"{package} is missing")
    
    if missing_packages:
        print(f"\n Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_file_structure():
    print("\n Checking file structure")
    
    required_structure = {
        'src/': ['tree_node.py', 'splitting_criteria.py', 'decision_tree.py', 'decision_functions.py'],
        'experiments/': ['step1_data_exploration.py', 'step2_multiple_criteria.py', 
                        'step3_stopping_criteria.py', 'step4_hyperparameter_tuning.py',
                        'step5_final_evaluation.py'],
        'data/': []
    }
    
    all_good = True
    
    for directory, files in required_structure.items():
        if not os.path.exists(directory):
            print(f"Directory missing: {directory}")
            if directory == 'data/':
                os.makedirs(directory, exist_ok=True)
                print(f"Created directory: {directory}")
            else:
                all_good = False
        else:
            print(f"Directory exists: {directory}")
        
        for file in files:
            filepath = os.path.join(directory, file)
            if not os.path.exists(filepath):
                print(f"File missing: {filepath}")
                all_good = False
            else:
                print(f"File exists: {filepath}")
    
    return all_good

def main():
    """
    Main script to run the complete decision tree analysis pipeline.
    """
    print("DECISION TREE MUSHROOM CLASSIFICATION ")
    print("=" * 60)
    
    #Check dependencies first
    if not check_dependencies():
        print("\nCannot proceed without required dependencies.")
        return
    
    #Check file structure
    if not check_file_structure():
        print("\n Cannot proceed with missing files.")
        return
    
    #Create results directory
    os.makedirs("results", exist_ok=True)
    
    #Define analysis steps
    steps = [
        ("Data Exploration & Preprocessing", "step1_data_exploration"),
        ("Multiple Splitting Criteria Analysis", "step2_multiple_criteria"), 
        ("Stopping Criteria Analysis", "step3_stopping_criteria"),
        ("Hyperparameter Tuning", "step4_hyperparameter_tuning"),
        ("Final Comprehensive Evaluation", "step5_final_evaluation")
    ]
    
    successful_steps = 0
    total_steps = len(steps)
    
    for step_name, module_name in steps:
        try:
            #Dynamic import of the step module
            module_path = f"experiments.{module_name}"
            module = __import__(module_path, fromlist=[''])
            
            #Find the main function to call
            if hasattr(module, 'main'):
                success = run_step_safely(step_name, module.main)
            elif hasattr(module, 'evaluate_splitting_criteria'):
                success = run_step_safely(step_name, module.evaluate_splitting_criteria)
            elif hasattr(module, 'evaluate_stopping_criteria'):
                success = run_step_safely(step_name, module.evaluate_stopping_criteria)
            elif hasattr(module, 'hyperparameter_tuning'):
                success = run_step_safely(step_name, module.hyperparameter_tuning)
            elif hasattr(module, 'final_comprehensive_evaluation'):
                success = run_step_safely(step_name, module.final_comprehensive_evaluation)
            else:
                print(f" No suitable function found in {module_name}")
                success = False
            
            if success:
                successful_steps += 1
                
        except ImportError as e:
            print(f" Could not import {module_path}: {str(e)}")
        except Exception as e:
            print(f" Unexpected error with {step_name}: {str(e)}")
    
    #Final summary
    print(f"\n{'='*60}")
    print(" PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f" Successful steps: {successful_steps}/{total_steps}")
    print(f" Failed steps: {total_steps - successful_steps}/{total_steps}")
    
    if successful_steps == total_steps:
        print("All analysis steps completed successfully!")
    elif successful_steps > 0:
        print("Some steps completed successfully, but there were issues.")
    else:
        print("No steps completed successfully.")
    

if __name__ == "__main__":
    main()