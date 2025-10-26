import configparser

def parse_args(path):
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    
    # Convert to dictionary
    config_dict = {}
    for section_name in parser.sections():
        config_dict[section_name] = dict(parser.items(section_name))
    
    # Also include DEFAULT section
    if parser.defaults():
        config_dict['DEFAULT'] = dict(parser.defaults())
    
    return config_dict