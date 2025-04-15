import torch
from torch import nn
import os
import copy

def export_policy_as_jit(actor_critic, path, exported_policy_name):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, exported_policy_name)
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

def export_policy_as_onnx(inference_model, path, exported_policy_name, example_obs_dict):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, exported_policy_name)

        actor = copy.deepcopy(inference_model['actor']).to('cpu')

        class PPOWrapper(nn.Module):
            def __init__(self, actor):
                """
                model: The original PyTorch model.
                input_keys: List of input names as keys for the input dictionary.
                """
                super(PPOWrapper, self).__init__()
                self.actor = actor

            def forward(self, actor_obs):
                """
                Dynamically creates a dictionary from the input keys and args.
                """
                return self.actor.act_inference(actor_obs)

        wrapper = PPOWrapper(actor)
        example_input_list = example_obs_dict["actor_obs"]
        torch.onnx.export(
            wrapper,
            example_input_list,  # Pass x1 and x2 as separate inputs
            path,
            verbose=True,
            input_names=["actor_obs"],  # Specify the input names
            output_names=["action"],       # Name the output
            opset_version=13           # Specify the opset version, if needed
        )

def export_policy_and_estimator_as_onnx(inference_model, path, exported_policy_name, example_obs_dict):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, exported_policy_name)

        actor = copy.deepcopy(inference_model['actor']).to('cpu')
        left_hand_force_estimator = copy.deepcopy(inference_model['left_hand_force_estimator']).to('cpu')
        right_hand_force_estimator = copy.deepcopy(inference_model['right_hand_force_estimator']).to('cpu')

        class PPOForceEstimatorWrapper(nn.Module):
            def __init__(self, actor, left_hand_force_estimator, right_hand_force_estimator):
                """
                model: The original PyTorch model.
                input_keys: List of input names as keys for the input dictionary.
                """
                super(PPOForceEstimatorWrapper, self).__init__()
                self.actor = actor
                self.left_hand_force_estimator = left_hand_force_estimator
                self.right_hand_force_estimator = right_hand_force_estimator

            def forward(self, inputs):
                """
                Dynamically creates a dictionary from the input keys and args.
                """
                actor_obs, history_for_estimator = inputs
                left_hand_force_estimator_output = self.left_hand_force_estimator(history_for_estimator)
                right_hand_force_estimator_output = self.right_hand_force_estimator(history_for_estimator)
                input_for_actor = torch.cat([actor_obs, left_hand_force_estimator_output, right_hand_force_estimator_output], dim=-1)
                return self.actor.act_inference(input_for_actor), left_hand_force_estimator_output, right_hand_force_estimator_output

        wrapper = PPOForceEstimatorWrapper(actor, left_hand_force_estimator, right_hand_force_estimator)
        example_input_list = [example_obs_dict["actor_obs"], example_obs_dict["long_history_for_estimator"]]
        torch.onnx.export(
            wrapper,
            example_input_list,  # Pass x1 and x2 as separate inputs
            path,
            verbose=True,
            input_names=["actor_obs", "long_history_for_estimator"],  # Specify the input names
            output_names=["action", "left_hand_force_estimator_output", "right_hand_force_estimator_output"],       # Name the output
            opset_version=13           # Specify the opset version, if needed
        )

def export_decouple_policy_and_estimator_history_as_onnx(inference_model, path, exported_policy_name, example_obs_dict):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, exported_policy_name)

        actor = copy.deepcopy(inference_model['actor']).to('cpu')
        left_ee_force_estimator = copy.deepcopy(inference_model['left_ee_force_estimator']).to('cpu')
        right_ee_force_estimator = copy.deepcopy(inference_model['right_ee_force_estimator']).to('cpu')

        class PPODecoupleForceEstimatorWrapper(nn.Module):
            def __init__(self, actor, left_ee_force_estimator, right_ee_force_estimator):
                """
                model: The original PyTorch model.
                input_keys: List of input names as keys for the input dictionary.
                """
                super(PPODecoupleForceEstimatorWrapper, self).__init__()
                self.actor = actor
                self.left_ee_force_estimator = left_ee_force_estimator
                self.right_ee_force_estimator = right_ee_force_estimator

            def forward(self, inputs):
                """
                Dynamically creates a dictionary from the input keys and args.
                """
                actor_obs, estimator_obs, history_estimated_force = inputs
                left_ee_force_estimator_output = self.left_ee_force_estimator(estimator_obs)
                right_ee_force_estimator_output = self.right_ee_force_estimator(estimator_obs)
                input_for_actor = torch.cat([actor_obs, 
                                             left_ee_force_estimator_output, 
                                             right_ee_force_estimator_output,
                                             history_estimated_force], dim=-1)
                return self.actor.act_inference(input_for_actor), left_ee_force_estimator_output, right_ee_force_estimator_output

        wrapper = PPODecoupleForceEstimatorWrapper(actor, left_ee_force_estimator, right_ee_force_estimator)
        example_input_list = [example_obs_dict["actor_obs"], 
                              example_obs_dict["estimator_obs"],
                              example_obs_dict["estimated_force_history"]]
        torch.onnx.export(
            wrapper,
            example_input_list,  # Pass x1 and x2 as separate inputs
            path,
            verbose=True,
            input_names=["actor_obs", "estimator_obs", "estimated_force_history"],  # Specify the input names
            output_names=["action", "left_ee_force_estimator_output", "right_ee_force_estimator_output"],       # Name the output
            opset_version=13           # Specify the opset version, if needed
        )

def export_decouple_policy_and_estimator_as_onnx(inference_model, path, exported_policy_name, example_obs_dict):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, exported_policy_name)

        actor = copy.deepcopy(inference_model['actor']).to('cpu')
        left_ee_force_estimator = copy.deepcopy(inference_model['left_ee_force_estimator']).to('cpu')
        right_ee_force_estimator = copy.deepcopy(inference_model['right_ee_force_estimator']).to('cpu')

        class PPODecoupleForceEstimatorWrapper(nn.Module):
            def __init__(self, actor, left_ee_force_estimator, right_ee_force_estimator):
                """
                model: The original PyTorch model.
                input_keys: List of input names as keys for the input dictionary.
                """
                super(PPODecoupleForceEstimatorWrapper, self).__init__()
                self.actor = actor
                self.left_ee_force_estimator = left_ee_force_estimator
                self.right_ee_force_estimator = right_ee_force_estimator

            def forward(self, inputs):
                """
                Dynamically creates a dictionary from the input keys and args.
                """
                actor_obs, estimator_obs = inputs
                left_ee_force_estimator_output = self.left_ee_force_estimator(estimator_obs)
                right_ee_force_estimator_output = self.right_ee_force_estimator(estimator_obs)
                input_for_actor = torch.cat([actor_obs, 
                                             left_ee_force_estimator_output, 
                                             right_ee_force_estimator_output], dim=-1)
                return self.actor.act_inference(input_for_actor), left_ee_force_estimator_output, right_ee_force_estimator_output

        wrapper = PPODecoupleForceEstimatorWrapper(actor, left_ee_force_estimator, right_ee_force_estimator)
        example_input_list = [example_obs_dict["actor_obs"], 
                              example_obs_dict["estimator_obs"]]
        torch.onnx.export(
            wrapper,
            example_input_list,  # Pass x1 and x2 as separate inputs
            path,
            verbose=True,
            input_names=["actor_obs", "estimator_obs"],  # Specify the input names
            output_names=["action", "left_ee_force_estimator_output", "right_ee_force_estimator_output"],       # Name the output
            opset_version=13           # Specify the opset version, if needed
        )
    
def export_multi_agent_decouple_policy_as_onnx(inference_model, path, exported_policy_name, example_obs_dict):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, exported_policy_name)

        actor_lower_body = copy.deepcopy(inference_model['actors']['lower_body']).to('cpu')
        actor_upper_body = copy.deepcopy(inference_model['actors']['upper_body']).to('cpu')

        class PPOMADecoupleWrapper(nn.Module):
            def __init__(self, actor_lower_body, actor_upper_body):
                """
                model: The original PyTorch model.
                input_keys: List of input names as keys for the input dictionary.
                """
                super(PPOMADecoupleWrapper, self).__init__()
                self.actor_lower_body = actor_lower_body
                self.actor_upper_body = actor_upper_body

            def forward(self, actor_obs):
                """
                Dynamically creates a dictionary from the input keys and args.
                """
                return torch.concat([self.actor_lower_body.act_inference(actor_obs),
                                     self.actor_upper_body.act_inference(actor_obs)], dim=-1)

        wrapper = PPOMADecoupleWrapper(actor_lower_body, actor_upper_body)
        example_input_list = example_obs_dict["actor_obs"]
        torch.onnx.export(
            wrapper,
            example_input_list,  # Pass x1 and x2 as separate inputs
            path,
            verbose=True,
            input_names=["actor_obs"],  # Specify the input names
            output_names=["action"],       # Name the output
            opset_version=13           # Specify the opset version, if needed
        )
