# Backend Administrative Scripts

This directory contains administrative scripts for managing the backend application.

## Create Admin User

Script to create or upgrade users to admin or super admin roles.

### Usage

**Inside Docker container (recommended):**

```bash
# Create regular admin user
docker-compose exec backend python scripts/create_admin.py <email> <password>

# Create super admin user
docker-compose exec backend python scripts/create_admin.py <email> <password> --super-admin
```

### Examples

```bash
# Create a regular admin user
docker-compose exec backend python scripts/create_admin.py admin@example.com SecurePass123

# Create a super admin user
docker-compose exec backend python scripts/create_admin.py superadmin@example.com SecurePass123 --super-admin

# Upgrade existing user to admin
docker-compose exec backend python scripts/create_admin.py user@example.com password123

# Upgrade existing admin to super admin
docker-compose exec backend python scripts/create_admin.py admin@example.com password123 --super-admin
```

### Notes

- If the user already exists, the script will upgrade their role
- Passwords are hashed using bcrypt before storage
- Requires the backend container to be running (`docker-compose up backend`)
