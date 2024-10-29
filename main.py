from connect_four_evolution import train_evolutionary, play_against_evolved_model
import argparse
import torch

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train and play Connect Four using evolutionary neural networks')
    
    # Add arguments
    parser.add_argument('--generations', type=int, default=50,
                        help='Number of generations to evolve (default: 50)')
    parser.add_argument('--population-size', type=int, default=30,
                        help='Size of the population in each generation (default: 30)')
    parser.add_argument('--games-per-match', type=int, default=5,
                        help='Number of games played between each pair of models (default: 5)')
    parser.add_argument('--save-model', action='store_true',
                        help='Save the best model after training')
    parser.add_argument('--load-model', type=str,
                        help='Path to a previously saved model to load instead of training')
    parser.add_argument('--player-number', type=int, choices=[1, 2], default=1,
                        help='Play as player 1 or 2 (default: 1)')

    args = parser.parse_args()

    if args.load_model:
        # Load a previously saved model
        from connect_four_evolution import ConnectFourNN
        print(f"Loading model from {args.load_model}")
        model = ConnectFourNN()
        model.load_state_dict(torch.load(args.load_model))
        best_model = model
    else:
        # Train a new model
        print("Starting evolutionary training...")
        print(f"Generations: {args.generations}")
        print(f"Population size: {args.population_size}")
        print(f"Games per match: {args.games_per_match}")
        
        best_model, fitness_history = train_evolutionary(
            generations=args.generations,
            population_size=args.population_size,
            games_per_match=args.games_per_match
        )

        if args.save_model:
            model_path = 'best_connect_four_model.pth'
            print(f"Saving model to {model_path}")
            torch.save(best_model.state_dict(), model_path)

    # Play against the model
    print("\nStarting game against AI...")
    print("Enter a column number (0-6) to make your move.")
    play_against_evolved_model(best_model, human_player=args.player_number)

if __name__ == '__main__':
    main()