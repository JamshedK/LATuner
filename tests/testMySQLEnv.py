#!/usr/bin/env python3
"""
Test script for MySQLEnv Docker integration
Tests the basic lifecycle: kill -> start -> get_db_size
"""

import time
import os
from MySQLEnv import MySQLEnv

def test_mysql_env():
    print("=" * 60)
    print("Testing MySQLEnv Docker Integration")
    print("=" * 60)
    
    # Create a simple Docker-compatible template if it doesn't exist
    template_content = """[mysqld]
        skip-name-resolve
        port = 3306
        character-set-server=utf8
        # Knobs will be appended here
        """
    
    if not os.path.exists("./template_docker.cnf"):
        with open("./template_docker.cnf", "w") as f:
            f.write(template_content)
        print("✓ Created template_docker.cnf")
    
    # Create MySQLEnv instance
    print("\n1. Creating MySQLEnv instance...")
    try:
        mysql_env = MySQLEnv(
            host='localhost',
            user='root', 
            passwd='',
            dbname='benchbase',
            workload='benchbase_tpcc_20_16',
            objective='tps',
            method='test',
            stress_test_duration=60,
            template_cnf_path='./template_docker.cnf',
            real_cnf_path='./my_docker.cnf'
        )
        print("✓ MySQLEnv instance created successfully")
    except Exception as e:
        print(f"✗ Failed to create MySQLEnv: {e}")
        return False
    
    # Test 1: Kill any existing MySQL
    print("\n2. Testing _kill_mysqld()...")
    try:
        mysql_env._kill_mysqld()
        print("✓ _kill_mysqld() completed")
        time.sleep(5)  # Wait for cleanup
    except Exception as e:
        print(f"✗ _kill_mysqld() failed: {e}")
        # Continue anyway, might not have been running
    
    # Test 2: Start MySQL
    print("\n3. Testing _start_mysqld()...")
    try:
        success = mysql_env._start_mysqld()
        if success:
            print("✓ _start_mysqld() completed successfully")
        else:
            print("✗ _start_mysqld() returned False")
            return False
    except Exception as e:
        print(f"✗ _start_mysqld() failed: {e}")
        return False
    
    # Wait a bit for MySQL to fully start
    print("\n4. Waiting for MySQL to stabilize...")
    time.sleep(10)
    
    # Test 3: Get database size
    print("\n5. Testing get_db_size()...")
    try:
        db_size = mysql_env.get_db_size()
        print(f"✓ Database size: {db_size} MB")
    except Exception as e:
        print(f"✗ get_db_size() failed: {e}")
        return False
    
    # Test 4: Test configuration replacement
    print("\n6. Testing replace_mycnf() with sample knobs...")
    try:
        test_knobs = {
            'innodb_buffer_pool_size': '128M',
            'max_connections': 200,
            'innodb_write_io_threads': 4
        }
        mysql_env.replace_mycnf(test_knobs)
        print("✓ Configuration replacement completed")
        
        # Show the generated config
        if os.path.exists(mysql_env.real_cnf_path):
            print(f"\nGenerated config file ({mysql_env.real_cnf_path}):")
            with open(mysql_env.real_cnf_path, 'r') as f:
                print(f.read())
    except Exception as e:
        print(f"✗ replace_mycnf() failed: {e}")
        return False
    
    # Test 5: Apply knobs (full restart cycle)
    print("\n7. Testing apply_knobs() (full restart cycle)...")
    try:
        test_knobs = {
            'innodb_buffer_pool_size': '64M',
            'max_connections': 102
        }
        success = mysql_env.apply_knobs(test_knobs)
        if success:
            print("✓ apply_knobs() completed successfully")
        else:
            print("✗ apply_knobs() returned False")
            return False
    except Exception as e:
        print(f"✗ apply_knobs() failed: {e}")
        return False
    
    # Final database size check
    print("\n8. Final database size check...")
    try:
        final_db_size = mysql_env.get_db_size()
        print(f"✓ Final database size: {final_db_size} MB")
    except Exception as e:
        print(f"✗ Final get_db_size() failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("MySQLEnv Docker integration is working correctly")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    print("Starting MySQLEnv Docker Integration Test")
    print("Make sure Docker is running and mysql57-latuner container is not already running")
    print("This test will create/remove Docker containers during testing")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    try:
        success = test_mysql_env()
        if success:
            print("\n🎉 Test completed successfully!")
        else:
            print("\n❌ Test failed!")
            exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        exit(1)
