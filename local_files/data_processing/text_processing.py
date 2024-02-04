class TextProcessing():
    index_to_char_map = {
        0: "'", 1: " ", 2: "a", 3: "b", 4: "c", 5: "d", 6: "e", 7: "f", 8: "g", 9: "h", 10: "i", 
        11: "j", 12: "k", 13: "l", 14: "m", 15: "n", 16: "o", 17: "p", 18: "q", 19: "r", 20: "s",
        21: "t", 22: "u", 23: "v", 24: "w", 25: "x", 26: "y", 27: "z", 28: "_", # blank character
    }
    char_to_index_map = { 
        "'": 0, " ": 1, "a": 2, "b": 3, "c": 4, "d": 5, "e": 6, "f": 7, "g": 8, "h": 9, "i": 10, 
        "j": 11, "k": 12, "l": 13, "m": 14, "n": 15, "o": 16, "p": 17, "q": 18, "r": 19, "s": 20, 
        "t": 21, "u": 22, "v": 23, "w": 24, "x": 25, "y": 26, "z": 27, "_": 28 # blank character
    }

    def text_to_int_sequence(text):
        int_sequence = []
        for char in text.lower():
            if char in TextProcessing.char_to_index_map.keys(): 
                index = TextProcessing.char_to_index_map[char]
            else: # Ignoring characters not specified in dictionary
                continue
            int_sequence.append(index)
        return int_sequence
    
    def int_sequence_to_text(int_sequence):
        text = ""
        for index in int_sequence:
            if index in TextProcessing.index_to_char_map.keys(): # Ignoring integers outside of range specified in dictionary
                text += TextProcessing.index_to_char_map[index]
        return text
    
    def text_with_only_allowed_characters(text):
        output_text = ""
        for char in text.lower():
            if char in TextProcessing.char_to_index_map.keys():
                output_text += char
        return output_text
    
    def get_char_list():
        return list(TextProcessing.char_to_index_map.keys())

    def get_index_list():
        return list(TextProcessing.index_to_char_map.keys())
    
print(TextProcessing.get_char_list)