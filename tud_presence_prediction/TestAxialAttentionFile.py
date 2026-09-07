import torch
import sys
import os

# Pfade anpassen, da wir uns bereits im tud_presence_prediction Ordner befinden
from tud_presence_prediction.models.LearnTransformerAxialSelf import (
    LearnTransformerEncoderBlock,
    LearnTransformerDecoderBlock,
    LearnTransformer
)

# Testfunktionen hier einfuegen...
def test_encoder_block():
    """Test fuer den Encoder-Block"""
    hidden_dim = 8
    block = LearnTransformerEncoderBlock(hidden_dim, num_heads=4, dropout=0.1, use_flash_attention=False)
    
    # Test mit Batch-Dimension
    x = torch.rand(2, 3, 4, hidden_dim*2)  # [batch, days, timeslots, features]
    output = block(x)
    assert output.shape == (2, 3, 4, hidden_dim*2), f"Expected shape {(2, 3, 4, hidden_dim*2)}, got {output.shape}"
    
    print("Encoder Block Test bestanden!")

def test_decoder_block():
    """Test fuer den Decoder-Block"""
    hidden_dim = 8
    print("Erstelle Bloecke")
    block = LearnTransformerDecoderBlock(hidden_dim, num_heads=4, dropout=0.1, use_flash_attention=False)
    
    # Test mit normalen Inputs
    x = torch.rand(2, 3, 4, hidden_dim*2)  # [batch, days, timeslots, features]
    enc_output = torch.rand(2, 3, 4, hidden_dim*2)
    print("Fuehre Block aus")
    output = block(x, enc_output)
    assert output.shape == (2, 3, 4, hidden_dim*2), f"Shape mismatch: {output.shape}"
    
    # Test mit Inference-Parametern
    output = block(x, enc_output)
    assert output.shape == (2, 3, 4, hidden_dim*2), f"Shape mismatch with t_cur/d_cur: {output.shape}"
    
    print("Decoder Block Test bestanden!")

def test_full_model(use_flash=False, step_size=1):
    """Test des vollstaendigen Modells"""
    model = LearnTransformer(
        location_dim=10, 
        calendar_dim=5, 
        hidden_dim=8, 
        num_heads=4, 
        dropout=0.1,
        step_size=step_size,
        use_flash_attention=use_flash
    )
    
    # Trainingsmodus
    batch_size, days, timeslots = 2, 3, 4
    x_loc = torch.rand(batch_size, days, timeslots, 10)
    x_cal = torch.rand(batch_size, days, timeslots, 5)
    target = torch.rand(batch_size, days, timeslots)
    
    # Training
    model.train()
    output = model(x_loc, x_cal, target)
    assert output.shape == (batch_size, days, timeslots), f"Training output shape mismatch: {output.shape}"
    
    # Inferenz
    model.eval()
    model.set_model_prediction_length(days)
    output = model(x_loc, x_cal)
    assert output.shape == (batch_size, days, timeslots), f"Inference output shape mismatch: {output.shape}"
    
    print(f"Vollstaendiger Modelltest mit use_flash={use_flash}, step_size={step_size} bestanden!")

def test_step_by_step_inference():
    """Testet die stufenweise Inferenz im Single-Step-Modus"""
    model = LearnTransformer(
        location_dim=10, 
        calendar_dim=5, 
        hidden_dim=8, 
        num_heads=4, 
        dropout=0.1,
        step_size=1,  # Single-step Modus
        use_flash_attention=False
    )
    
    batch_size, days, timeslots = 1, 2, 3
    x_loc = torch.rand(batch_size, days, timeslots, 10)
    x_cal = torch.rand(batch_size, days, timeslots, 5)
    
    model.eval()
    model.set_model_prediction_length(days)
    
    # Debug-Ausgabe aktivieren für ein besseres Verständnis
    print("\n=== Stufenweise Inferenztest ===")
    output = model(x_loc, x_cal)
    print(f"Ausgabeform: {output.shape}")
    print("Stufenweise Inferenztest bestanden!")

if __name__ == "__main__":
    print("Starte Tests...")
    try:
        test_encoder_block()
        test_decoder_block()
        
        # Test einzelner Modi zunächst getrennt für bessere Fehlerfindung
        print("\n--- Teste Standard-Attention, single-step ---")
        test_full_model(use_flash=False, step_size=1)
        
        print("\n--- Teste Flash-Attention, single-step ---")
        test_full_model(use_flash=True, step_size=1)
        
        # print("\n--- Teste multi-step Inferenz ---")
        # test_full_model(use_flash=False, step_size=2)
        
        # Zusätzlicher Test für stufenweise Vorhersage
        test_step_by_step_inference()
        
        print("\nAlle Tests erfolgreich abgeschlossen!")
    except Exception as e:
        print(f"Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()