import os

queries_dir = './queries_out'
output_file = './tpch_4400.sql'

with open(output_file, 'w') as out:
    for i in range(1, 23):
        file_path = os.path.join(queries_dir, f'{i}.sql')
        with open(file_path, 'r') as f:
            query = f.read()
            # Remove seed comments
            lines = [line for line in query.split('\n') if not line.strip().startswith('-- using')]
            query = '\n'.join(lines)
            # Remove all whitespace and newlines, then collapse multiple spaces
            query = ' '.join(query.split())
            out.write(query + '\n')

print(f"Created {output_file} with 22 queries (each on single line)")
