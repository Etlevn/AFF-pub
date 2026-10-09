def plot_loss_ascii(train_losses, valid_losses=None, title="Loss Plot", width=60, height=20):
    """
    Draw the loss curve graph of ASCII format in the terminal (optimized version)

    Args:
        train_losses: Training loss list
        valid_losses: Validation loss list (optional)
        title: Chart title
        width: chart width (number of characters, default 60 for better point density)
        height: chart height (number of characters, default 20 for higher accuracy)
    """
    if not train_losses:
        print("  [Plot] No loss data to draw")
        return

    # Prepare data
    epochs = list(range(1, len(train_losses) + 1))
    all_losses = train_losses[:]
    if valid_losses:
        all_losses.extend(valid_losses)

    # Calculate data range
    min_loss = min(all_losses)
    max_loss = max(all_losses)
    loss_range = max_loss - min_loss
    if loss_range == 0:
        loss_range = 1  # Avoid division by zero

    min_epoch = min(epochs)
    max_epoch = max(epochs)
    epoch_range = max_epoch - min_epoch
    if epoch_range == 0:
        epoch_range = 1  # Avoid division by zero

    print(f"\n  {title}")
    print("  " + "=" * width)

    # Draw charts
    for row in range(height):
        line = "  "
        # Y axis label (fixed width alignment)
        y_val = max_loss - (row / (height - 1)) * loss_range
        line += f"{y_val:8.4f}│"

        # Draw data points (draw directly in epoch order to ensure that each data point is displayed)
        for col in range(width - 9):  # Consistent with X axis label width
            char = " "

            # Calculate the epoch position corresponding to the current column
            if len(epochs) > 1:
                # Map column position to epoch index
                epoch_float = col * (len(epochs) - 1) / (width - 10)
                epoch_idx = int(round(epoch_float))

                # Ensure that the index is within the valid range
                if 0 <= epoch_idx < len(train_losses):
                    # Check training loss
                    train_y = (max_loss - train_losses[epoch_idx]) / loss_range * (height - 1)
                    if abs(train_y - row) < 0.5:  # Slightly relax the matching threshold
                        char = "*"

                # Check validation loss
                if valid_losses and 0 <= epoch_idx < len(valid_losses):
                    valid_y = (max_loss - valid_losses[epoch_idx]) / loss_range * (height - 1)
                    if abs(valid_y - row) < 0.5:  # Slightly relax the matching threshold
                        if char == "*":
                            char = "▲"  # Training and validation losses overlap
                        else:
                            char = "+"
            else:
                # The case where there is only one epoch
                if col == (width - 10) // 2:  # middle position
                    train_y = (max_loss - train_losses[0]) / loss_range * (height - 1)
                    if abs(train_y - row) < 0.5:
                        char = "*"

                    if valid_losses:
                        valid_y = (max_loss - valid_losses[0]) / loss_range * (height - 1)
                        if abs(valid_y - row) < 0.5:
                            if char == "*":
                                char = "▲"
                            else:
                                char = "+"

            line += char

        print(line)

    # X axis (repair Y axis intersection alignment)
    print("  " + " " * 8 + "└" + "─" * (width - 9))  # Consistent with Y axis label width

    # X axis label (fix alignment and correspondence)
    x_labels = "  " + " " * 8  # Consistent with Y axis label width
    label_positions = []
    for i in range(0, width - 9, max(1, (width - 9) // 5)):  # Adjust width calculation
        x_val = min_epoch + (i / (width - 10)) * epoch_range  # Adjust width calculation
        label_positions.append((i, int(x_val)))

    # Ensure that the first and last tags correspond correctly
    if label_positions:
        # The first tag should be the first epoch
        label_positions[0] = (0, min_epoch)
        # The last tag should be the last epoch
        if len(label_positions) > 1:
            label_positions[-1] = (width - 9, max_epoch)  # Adjust width

    # Construct label string
    for i, (pos, epoch_val) in enumerate(label_positions):
        if i == 0:
            x_labels += f"{epoch_val:>6}"
        else:
            # Calculate the distance from the previous label
            prev_pos = label_positions[i-1][0]
            spacing = pos - prev_pos
            x_labels += " " * (spacing - 6) + f"{epoch_val:>6}"

    print(x_labels)
    print("  " + " " * 8 + "Epochs")  # Consistent with Y axis label width

    # Legend
    legend = "  Legend: "
    if valid_losses:
        legend += "* Train Loss, + Valid Loss, ▲ Overlap"
    else:
        legend += "* Train Loss"
    print(legend)

    # Statistical information
    print(f"  Final Train Loss: {train_losses[-1]:.5f}")
    if valid_losses:
        print(f"  Final Valid Loss: {valid_losses[-1]:.5f}")

    # Additional statistical information (showing loss trends)
    if len(train_losses) > 1:
        train_improvement = train_losses[0] - train_losses[-1]
        print(f"  Train Loss Improvement: {train_improvement:+.5f}")
        if valid_losses and len(valid_losses) > 1:
            valid_improvement = valid_losses[0] - valid_losses[-1]
            print(f"  Valid Loss Improvement: {valid_improvement:+.5f}")

    # Display loss range
    print(f"  Train Loss Range: [{min(train_losses):.5f}, {max(train_losses):.5f}]")
    if valid_losses:
        print(f"  Valid Loss Range: [{min(valid_losses):.5f}, {max(valid_losses):.5f}]")

    print("  " + "=" * width)
    print()


def plot_generator_loss_ascii(losses, title="Generator Loss Plot", width=60, height=20):
    """
    Draw the generator loss curve graph of ASCII format in the terminal (optimized version)

    Args:
        losses: Loss list
        title: Chart title
        width: chart width (number of characters, default 60 for better point density)
        height: chart height (number of characters, default 20 for higher accuracy)
    """
    plot_loss_ascii(losses, None, title, width, height)
