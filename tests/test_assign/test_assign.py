import json
from pathlib import Path
from io_models import UserOutput
from assign import Args, main
from utils.types import UserInputPath


def get_test_user_input_path(name: str):
    return UserInputPath(f"./tests/test_assign/{name}/user_input.json")

def get_test_user_output_path(name: str):
    return Path(f"./tests/test_assign/{name}/user_output.json")



def test_impossible_input():
    user_outputs = main(
        Args(user_input_json_path=get_test_user_input_path("impossible"), store=False)
    )
    
    with open(get_test_user_output_path("impossible")) as f:
        correct_user_outputs = UserOutput.model_validate(json.load(f))
        
    assert user_outputs == correct_user_outputs
    
def test_normal_input():
    user_outputs = main(
        Args(user_input_json_path=get_test_user_input_path("normal"), store=False)
    )
    
    with open(get_test_user_output_path("normal")) as f:
        correct_user_outputs = UserOutput.model_validate(json.load(f))
        
    assert user_outputs == correct_user_outputs
    
    
def test_trivial_input():
    user_outputs = main(
        Args(user_input_json_path=get_test_user_input_path("trivial"), store=False)
    )
    
    with open(get_test_user_output_path("trivial")) as f:
        correct_user_outputs = UserOutput.model_validate(json.load(f))
        
    assert user_outputs == correct_user_outputs
