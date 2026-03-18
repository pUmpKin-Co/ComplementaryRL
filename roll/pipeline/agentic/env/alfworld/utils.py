import json
import os
import sys
import time

import numpy as np
import yaml
from tqdm import tqdm

TASK_TYPES = {1: "pick_and_place_simple",
              2: "look_at_obj_in_light",
              3: "pick_clean_then_place_in_recep",
              4: "pick_heat_then_place_in_recep",
              5: "pick_cool_then_place_in_recep",
              6: "pick_two_obj_and_place"}


def retry_file_operation(func, max_retries=5, initial_delay=1.0, backoff_factor=2, max_delay=30):
    """
    Retry a file operation with exponential backoff to handle transient filesystem issues.
    
    Args:
        func: Callable that performs the file operation
        max_retries: Maximum number of retry attempts (default: 5)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        backoff_factor: Multiplier for delay between retries (default: 2)
        max_delay: Maximum delay between retries in seconds (default: 30)
    
    Returns:
        Result of func() if successful
    
    Raises:
        OSError: If all retries fail
    """
    last_exception = None
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return func()
        except OSError as e:
            last_exception = e
            # Only retry on specific transient errors
            if e.errno in (107,  # Transport endpoint is not connected
                          116,  # Stale file handle
                          5,    # Input/output error
                          110): # Connection timed out
                if attempt < max_retries - 1:
                    # Use stderr to avoid potential logging issues
                    msg = f"[OSS Mount Warning] File operation failed: {e}. Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...\n"
                    sys.stderr.write(msg)
                    sys.stderr.flush()
                    time.sleep(delay)
                    # Cap the delay at max_delay
                    delay = min(delay * backoff_factor, max_delay)
                else:
                    # Last attempt failed
                    msg = f"[OSS Mount Error] All {max_retries} retry attempts failed. Last error: {e}\n"
                    sys.stderr.write(msg)
                    sys.stderr.flush()
                    raise
            else:
                # For non-transient errors, fail immediately
                raise
        except Exception as e:
            # For non-OSError exceptions, fail immediately
            raise
    
    # This should not be reached, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected state in retry_file_operation")


def collect_game_files(train_eval, env_config, data_path, labeled_data, verbose=False):
    def log(info):
        if verbose:
            print(info)
    
    with open(env_config) as reader:
        config = yaml.safe_load(reader)

    game_files = []
    skipped_files = []
    
    log("Collecting solvable games...")
    # get task types
    assert len(config['env']['task_types']) > 0
    task_types = []
    for tt_id in config['env']['task_types']:
        if tt_id in TASK_TYPES:
            task_types.append(TASK_TYPES[tt_id])
    # print(labeled_data)
    count = 0
    # print(data_path)
    for root, dirs, files in tqdm(list(os.walk(data_path, topdown=False))):
     
        if root not in labeled_data:
            # print(1)
            continue

        if 'traj_data.json' in files:
            count += 1
            # print(1)
            # Filenames
            json_path = os.path.join(root, 'traj_data.json')
            game_file_path = os.path.join(root, "game.tw-pddl")
          
            if 'movable' in root or 'Sliced' in root:
                log("Movable & slice trajs not supported %s" % (root))
                continue

            try:
                # Get goal description
                def read_traj_data():
                    with open(json_path, 'r') as f:
                        return json.load(f)
                
                traj_data = retry_file_operation(read_traj_data)

                # Check for any task_type constraints
                if not traj_data['task_type'] in task_types:
                    log("Skipping task type")
                    continue

                # Check if a game file exists
                if not os.path.exists(game_file_path):
                    log(f"Skipping missing game! {game_file_path}")
                    continue

                def read_game_data():
                    with open(game_file_path, 'r') as f:
                        return json.load(f)
                
                gamedata = retry_file_operation(read_game_data)

                # Check if previously checked if solvable
                if 'solvable' not in gamedata:
                    print(f"-> Skipping missing solvable key! {game_file_path}")
                    continue

                if not gamedata['solvable']:
                    log("Skipping known %s, unsolvable game!" % game_file_path)
                    continue

                # Add to game file list
                game_files.append(game_file_path)
                
            except (OSError, IOError) as e:
                # Skip files that are persistently inaccessible due to OSS mount issues
                msg = f"[OSS Mount Error] Skipping inaccessible file {json_path}: {e}\n"
                sys.stderr.write(msg)
                sys.stderr.flush()
                skipped_files.append(json_path)
                continue
            except Exception as e:
                # Log unexpected errors but continue processing
                msg = f"[Unexpected Error] Failed to process {json_path}: {e}\n"
                sys.stderr.write(msg)
                sys.stderr.flush()
                continue

    print(f"Overall we have {len(game_files)} games in split={train_eval}")
    if skipped_files:
        msg = f"[OSS Mount Warning] Skipped {len(skipped_files)} inaccessible files due to OSS mount issues\n"
        sys.stderr.write(msg)
        sys.stderr.flush()
    
    num_games = len(game_files)

    if train_eval == "train":
        # num_train_games = config['dataset']['num_train_games'] if config['dataset']['num_train_games'] > 0 else len(game_files)
        # game_files = game_files[:num_train_games]
        # num_games = len(game_files)
        print("Training with %d games" % (len(game_files)))
    else:
        # num_eval_games = config['dataset']['num_eval_games'] if config['dataset']['num_eval_games'] > 0 else len(game_files)
        # game_files = game_files[:num_eval_games]
        # num_games = len(game_files)
        print("Evaluating with %d games" % (len(game_files)))
    
    if len(game_files) == 0:
        raise RuntimeError(
            f"No accessible game files found in {data_path}. "
            f"This may indicate a complete OSS mount failure. "
            f"Skipped {len(skipped_files)} files. Please check mount status."
        )
    
    return sorted(game_files)

