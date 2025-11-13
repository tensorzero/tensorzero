# Remix Auth Integration Report for TensorZero UI

## Executive Summary

The TensorZero UI currently has **no authentication system**. This report outlines how to integrate `remix-auth` into the existing React Router v7 application to add user authentication.

## Current State

### What Exists
- ✅ React Router v7 with SSR enabled
- ✅ Middleware support (currently used for read-only mode)
- ✅ Context provider pattern (ReadOnlyProvider, ConfigProvider)
- ✅ Server-side utilities with `.server.ts` files
- ✅ PostgreSQL database with API key management
- ✅ Form handling with React Router's `Form` component
- ✅ TypeScript with strong typing throughout

### What's Missing
- ❌ No session management
- ❌ No user authentication
- ❌ No login/logout routes
- ❌ No protected routes
- ❌ No user context or user type definitions

## Recommended Architecture

### 1. Session Management

Create a cookie-based session storage system:

```typescript
// app/utils/session.server.ts
import { createCookieSessionStorage } from "react-router";

export const sessionStorage = createCookieSessionStorage({
  cookie: {
    name: "__tensorzero_session",
    httpOnly: true,
    path: "/",
    sameSite: "lax",
    secrets: [process.env.SESSION_SECRET!], // Add to environment variables
    secure: process.env.NODE_ENV === "production",
    maxAge: 60 * 60 * 24 * 7, // 7 days
  },
});

export const { getSession, commitSession, destroySession } = sessionStorage;
```

**Environment Variable Needed:**
```bash
SESSION_SECRET=your-random-secret-here  # Generate with: openssl rand -base64 32
```

### 2. User Type Definition

Define the user type that will be stored in the session:

```typescript
// app/types/user.ts
export interface User {
  id: string;
  email: string;
  name: string;
  role: "admin" | "user" | "viewer";
  organizationId: string;
  createdAt: string;
}
```

### 3. Authenticator Setup

Create the Remix Auth authenticator instance:

```typescript
// app/utils/auth.server.ts
import { Authenticator } from "remix-auth";
import { FormStrategy } from "remix-auth-form";
import { sessionStorage } from "./session.server";
import type { User } from "~/types/user";

// Create an instance of the authenticator
export const authenticator = new Authenticator<User>(sessionStorage);

// Register the form strategy
authenticator.use(
  new FormStrategy(async ({ form }) => {
    const email = form.get("email") as string;
    const password = form.get("password") as string;

    if (!email || !password) {
      throw new Error("Email and password are required");
    }

    // TODO: Implement your actual authentication logic
    // This would typically:
    // 1. Query PostgreSQL for user by email
    // 2. Verify password hash using bcrypt
    // 3. Return user object or throw error

    const user = await verifyCredentials(email, password);
    return user;
  }),
  "user-pass"
);

// Helper function to verify credentials
async function verifyCredentials(email: string, password: string): Promise<User> {
  // TODO: Replace with actual database query
  // const postgresClient = await getPostgresClient();
  // const user = await postgresClient.getUserByEmail(email);
  //
  // if (!user || !(await bcrypt.compare(password, user.passwordHash))) {
  //   throw new Error("Invalid email or password");
  // }
  //
  // return user;

  throw new Error("Authentication not implemented");
}

// Helper to check if user is authenticated
export async function requireAuth(request: Request): Promise<User> {
  const user = await authenticator.isAuthenticated(request);
  if (!user) {
    throw redirect("/auth/login");
  }
  return user;
}

// Helper to get optional user (doesn't redirect)
export async function getUser(request: Request): Promise<User | null> {
  return await authenticator.isAuthenticated(request);
}
```

### 4. Authentication Routes

#### Login Route

```typescript
// app/routes/auth/login.tsx
import { Form, data, redirect, Link } from "react-router";
import { authenticator } from "~/utils/auth.server";
import type { Route } from "./+types";

export async function loader({ request }: Route.LoaderArgs) {
  // If already authenticated, redirect to dashboard
  const user = await authenticator.isAuthenticated(request);
  if (user) {
    return redirect("/");
  }
  return null;
}

export async function action({ request }: Route.ActionArgs) {
  try {
    // Authenticate the user
    const user = await authenticator.authenticate("user-pass", request);

    // Get session and set user
    const session = await getSession(request.headers.get("cookie"));
    session.set(authenticator.sessionKey, user);

    // Redirect to home with session cookie
    return redirect("/", {
      headers: {
        "Set-Cookie": await commitSession(session),
      },
    });
  } catch (error) {
    // Handle authentication errors
    if (error instanceof Error) {
      return data(
        { error: error.message },
        { status: 401 }
      );
    }
    throw error;
  }
}

export default function Login({ actionData }: Route.ComponentProps) {
  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="w-full max-w-md space-y-8 rounded-lg border bg-card p-8">
        <div className="text-center">
          <h1 className="text-2xl font-bold">Sign in to TensorZero</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Enter your credentials to access the dashboard
          </p>
        </div>

        {actionData?.error && (
          <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
            {actionData.error}
          </div>
        )}

        <Form method="post" className="space-y-6">
          <div>
            <label htmlFor="email" className="block text-sm font-medium">
              Email
            </label>
            <input
              type="email"
              name="email"
              id="email"
              required
              className="mt-1 block w-full rounded-md border px-3 py-2"
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium">
              Password
            </label>
            <input
              type="password"
              name="password"
              id="password"
              required
              autoComplete="current-password"
              className="mt-1 block w-full rounded-md border px-3 py-2"
            />
          </div>

          <button
            type="submit"
            className="w-full rounded-md bg-primary px-4 py-2 text-primary-foreground hover:bg-primary/90"
          >
            Sign In
          </button>
        </Form>
      </div>
    </div>
  );
}
```

#### Logout Route

```typescript
// app/routes/auth/logout.tsx
import { redirect } from "react-router";
import { authenticator } from "~/utils/auth.server";
import type { Route } from "./+types";

export async function action({ request }: Route.ActionArgs) {
  return await authenticator.logout(request, { redirectTo: "/auth/login" });
}

export async function loader() {
  // Redirect GET requests to home
  return redirect("/");
}
```

### 5. User Context Provider

Create a context to make user data available throughout the app:

```typescript
// app/context/user.tsx
import { createContext, use, type ReactNode } from "react";
import type { User } from "~/types/user";

const UserContext = createContext<User | null>(null);

export function useUser(): User {
  const user = use(UserContext);
  if (!user) {
    throw new Error("useUser must be used within a UserProvider and user must be authenticated");
  }
  return user;
}

export function useOptionalUser(): User | null {
  return use(UserContext);
}

interface UserProviderProps {
  children: ReactNode;
  user: User | null;
}

export function UserProvider({ children, user }: UserProviderProps) {
  return <UserContext value={user}>{children}</UserContext>;
}
```

### 6. Root Loader & Layout Updates

Update the root loader to fetch the current user:

```typescript
// app/root.tsx
import { getUser } from "~/utils/auth.server";
import { UserProvider } from "~/context/user";

export async function loader({ request }: Route.LoaderArgs) {
  // Fetch current user (null if not authenticated)
  const user = await getUser(request);

  // ... existing loader logic ...

  return {
    user,
    // ... existing return values ...
  };
}

export function Layout({ children, loaderData }: Route.ComponentProps) {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <Links />
      </head>
      <body>
        <UserProvider user={loaderData?.user ?? null}>
          <ReadOnlyProvider value={loaderData?.isReadOnly ?? false}>
            <ConfigProvider config={loaderData?.config}>
              <ReactQueryProvider>
                {children}
              </ReactQueryProvider>
            </ConfigProvider>
          </ReadOnlyProvider>
        </UserProvider>
        <ScrollRestoration />
        <Scripts />
      </body>
    </html>
  );
}
```

### 7. Authentication Middleware

Create middleware to protect routes:

```typescript
// app/utils/auth-middleware.server.ts
import type { Route } from "react-router";
import { redirect } from "react-router";
import { getUser } from "./auth.server";

// Public routes that don't require authentication
const PUBLIC_ROUTES = [
  "/auth/login",
  "/auth/logout",
];

export const authMiddleware: Route.MiddlewareFunction = async ({ request }) => {
  const url = new URL(request.url);
  const pathname = url.pathname;

  // Skip auth check for public routes
  if (PUBLIC_ROUTES.some(route => pathname.startsWith(route))) {
    return;
  }

  // Check if user is authenticated
  const user = await getUser(request);

  // Redirect to login if not authenticated
  if (!user) {
    const searchParams = new URLSearchParams([
      ["returnTo", pathname + url.search],
    ]);
    throw redirect(`/auth/login?${searchParams}`);
  }
};
```

Update `root.tsx` to include auth middleware:

```typescript
// app/root.tsx
import { authMiddleware } from "~/utils/auth-middleware.server";
import { readOnlyMiddleware } from "~/utils/read-only.server";

export const middleware: Route.MiddlewareFunction[] = [
  authMiddleware,
  readOnlyMiddleware,
];
```

### 8. Protected Route Example

Example of protecting a specific route:

```typescript
// app/routes/api-keys/route.tsx
import { requireAuth } from "~/utils/auth.server";

export async function loader({ request }: Route.LoaderArgs) {
  // Require authentication for this route
  const user = await requireAuth(request);

  // Check permissions (optional)
  if (user.role === "viewer") {
    throw data(
      { error: "You don't have permission to view API keys" },
      { status: 403 }
    );
  }

  // ... rest of loader logic ...
}
```

### 9. UI Components for Auth

#### User Menu Component

```typescript
// app/components/layout/user-menu.tsx
import { Form } from "react-router";
import { useUser } from "~/context/user";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "~/components/ui/dropdown-menu";
import { Button } from "~/components/ui/button";

export function UserMenu() {
  const user = useUser();

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm">
          {user.name}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuLabel>
          <div className="flex flex-col space-y-1">
            <p className="text-sm font-medium">{user.name}</p>
            <p className="text-xs text-muted-foreground">{user.email}</p>
          </div>
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <Form method="post" action="/auth/logout">
          <DropdownMenuItem asChild>
            <button type="submit" className="w-full">
              Log out
            </button>
          </DropdownMenuItem>
        </Form>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
```

Add to your existing layout (e.g., `app/components/layout/header.tsx`):

```typescript
import { UserMenu } from "./user-menu";
import { useOptionalUser } from "~/context/user";

export function Header() {
  const user = useOptionalUser();

  return (
    <header>
      {/* ... existing header content ... */}
      {user && <UserMenu />}
    </header>
  );
}
```

## Database Schema Changes

You'll need to add a users table to PostgreSQL:

```sql
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  name VARCHAR(255) NOT NULL,
  role VARCHAR(50) NOT NULL DEFAULT 'user',
  organization_id UUID,
  created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
  disabled_at TIMESTAMPTZ
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_organization ON users(organization_id);
```

Add methods to your PostgreSQL client:

```typescript
// In tensorzero-node/src/postgres/mod.rs or equivalent
impl PostgresClient {
  pub async fn create_user(
    &self,
    email: &str,
    password_hash: &str,
    name: &str,
    role: &str,
  ) -> Result<User> {
    // Implementation
  }

  pub async fn get_user_by_email(&self, email: &str) -> Result<Option<User>> {
    // Implementation
  }

  pub async fn update_user(&self, user_id: &str, updates: UserUpdates) -> Result<User> {
    // Implementation
  }

  pub async fn disable_user(&self, user_id: &str) -> Result<()> {
    // Implementation
  }
}
```

## Dependencies to Install

```bash
npm install remix-auth remix-auth-form bcryptjs
npm install -D @types/bcryptjs
```

## Alternative Authentication Strategies

### OAuth 2.0 (GitHub, Google, etc.)

For OAuth authentication, use the OAuth2 strategy:

```bash
npm install remix-auth-oauth2
# Or use provider-specific strategies:
npm install remix-auth-github
npm install remix-auth-google
```

Example GitHub OAuth setup:

```typescript
// app/utils/auth.server.ts
import { GitHubStrategy } from "remix-auth-github";

authenticator.use(
  new GitHubStrategy(
    {
      clientId: process.env.GITHUB_CLIENT_ID!,
      clientSecret: process.env.GITHUB_CLIENT_SECRET!,
      redirectURI: `${process.env.APP_URL}/auth/github/callback`,
    },
    async ({ profile }) => {
      // Create or find user in database
      const user = await findOrCreateUser({
        email: profile.emails[0].value,
        name: profile.displayName,
        githubId: profile.id,
      });
      return user;
    }
  ),
  "github"
);
```

Callback route:

```typescript
// app/routes/auth/github/callback.tsx
import { authenticator } from "~/utils/auth.server";
import type { Route } from "./+types";

export async function loader({ request }: Route.LoaderArgs) {
  return await authenticator.authenticate("github", request, {
    successRedirect: "/",
    failureRedirect: "/auth/login",
  });
}
```

### Magic Link / Passwordless

For magic link authentication:

```bash
npm install remix-auth-email-link
```

This sends a one-time login link to the user's email.

## Implementation Checklist

### Phase 1: Core Setup
- [ ] Add `SESSION_SECRET` to environment variables
- [ ] Install dependencies: `remix-auth`, `remix-auth-form`, `bcryptjs`
- [ ] Create session storage (`app/utils/session.server.ts`)
- [ ] Define User type (`app/types/user.ts`)
- [ ] Create authenticator (`app/utils/auth.server.ts`)

### Phase 2: Database
- [ ] Add users table migration
- [ ] Add PostgreSQL methods for user CRUD
- [ ] Add TypeScript bindings for User type
- [ ] Create user seeding script for development

### Phase 3: Routes
- [ ] Create login route (`app/routes/auth/login.tsx`)
- [ ] Create logout route (`app/routes/auth/logout.tsx`)
- [ ] Add OAuth callback routes if using OAuth
- [ ] Test authentication flows

### Phase 4: Context & Middleware
- [ ] Create UserProvider (`app/context/user.tsx`)
- [ ] Update root loader to fetch user
- [ ] Add UserProvider to root layout
- [ ] Create auth middleware (`app/utils/auth-middleware.server.ts`)
- [ ] Add middleware to root

### Phase 5: UI Components
- [ ] Create UserMenu component
- [ ] Add UserMenu to header/layout
- [ ] Update navigation based on auth state
- [ ] Add role-based UI rendering

### Phase 6: Permissions
- [ ] Define role-based permissions
- [ ] Add permission checks to routes
- [ ] Update ReadOnlyGuard to consider user roles
- [ ] Add permission helpers

## Integration with Existing Features

### Read-Only Mode

You can combine authentication with the existing read-only mode:

```typescript
// app/utils/permissions.server.ts
import { useUser } from "~/context/user";
import { useReadOnly } from "~/context/read-only";

export function useCanWrite(): boolean {
  const user = useUser();
  const isReadOnly = useReadOnly();

  // Viewers can never write
  if (user.role === "viewer") return false;

  // Read-only mode prevents all writes
  if (isReadOnly) return false;

  return true;
}
```

### API Keys

Link API keys to users:

```sql
ALTER TABLE api_keys ADD COLUMN created_by_user_id UUID REFERENCES users(id);
ALTER TABLE api_keys ADD COLUMN organization_id UUID;
```

Update API key creation to track the creator:

```typescript
export async function action({ request }: Route.ActionArgs) {
  const user = await requireAuth(request);
  const formData = await request.formData();

  if (actionType === "generate") {
    const postgresClient = await getPostgresClient();
    const apiKey = await postgresClient.createApiKey(
      description,
      user.id,  // Track who created the key
      user.organizationId
    );
    return { apiKey };
  }
}
```

## Security Considerations

### 1. Password Storage
- ✅ Use bcrypt with salt rounds ≥ 10
- ✅ Never store plaintext passwords
- ✅ Use parameterized queries to prevent SQL injection

### 2. Session Security
- ✅ Set `httpOnly: true` on session cookies
- ✅ Use `secure: true` in production (HTTPS only)
- ✅ Set appropriate `maxAge` for sessions
- ✅ Use strong, random session secrets

### 3. CSRF Protection
- ✅ React Router's Form component includes CSRF protection
- ✅ Ensure all mutations use POST/PUT/DELETE (not GET)

### 4. Rate Limiting
Consider adding rate limiting for login attempts:

```typescript
// Pseudocode - implement with Redis or in-memory store
const loginAttempts = new Map<string, number>();

export async function action({ request }: Route.ActionArgs) {
  const email = (await request.formData()).get("email");

  if (loginAttempts.get(email) > 5) {
    throw data(
      { error: "Too many login attempts. Please try again later." },
      { status: 429 }
    );
  }

  try {
    const user = await authenticator.authenticate("user-pass", request);
    loginAttempts.delete(email); // Clear on success
    // ... rest of logic
  } catch (error) {
    loginAttempts.set(email, (loginAttempts.get(email) || 0) + 1);
    throw error;
  }
}
```

### 5. Permission Checks
Always check permissions on both client and server:

```typescript
// Server-side (required)
export async function action({ request }: Route.ActionArgs) {
  const user = await requireAuth(request);
  if (user.role !== "admin") {
    throw data({ error: "Unauthorized" }, { status: 403 });
  }
  // ... perform admin action
}

// Client-side (UX only)
function AdminButton() {
  const user = useUser();
  if (user.role !== "admin") return null;
  return <Button>Admin Action</Button>;
}
```

## Testing Strategy

### Unit Tests

Test authenticator functions:

```typescript
describe("authenticator", () => {
  it("should authenticate valid credentials", async () => {
    const user = await verifyCredentials("test@example.com", "password123");
    expect(user).toHaveProperty("id");
    expect(user.email).toBe("test@example.com");
  });

  it("should reject invalid credentials", async () => {
    await expect(
      verifyCredentials("test@example.com", "wrongpassword")
    ).rejects.toThrow("Invalid email or password");
  });
});
```

### Integration Tests

Test authentication flows:

```typescript
describe("/auth/login", () => {
  it("should redirect to home on successful login", async () => {
    const response = await fetch("/auth/login", {
      method: "POST",
      body: new URLSearchParams({
        email: "test@example.com",
        password: "password123",
      }),
    });

    expect(response.status).toBe(302);
    expect(response.headers.get("location")).toBe("/");
    expect(response.headers.get("set-cookie")).toContain("__tensorzero_session");
  });
});
```

### E2E Tests

Test complete user flows with Playwright or similar:

```typescript
test("user can log in and access protected routes", async ({ page }) => {
  await page.goto("/auth/login");
  await page.fill('input[name="email"]', "test@example.com");
  await page.fill('input[name="password"]', "password123");
  await page.click('button[type="submit"]');

  await expect(page).toHaveURL("/");
  await expect(page.locator("text=test@example.com")).toBeVisible();
});
```

## Migration Strategy

If deploying to an existing system:

1. **Phase 1**: Add authentication infrastructure (no enforcement)
   - Deploy code with auth routes and middleware
   - Keep middleware permissive (allow unauthenticated access)
   - Allow users to create accounts

2. **Phase 2**: Encourage adoption
   - Add banners encouraging users to create accounts
   - Show benefits (audit logs, personalization, etc.)
   - Provide migration path for existing data

3. **Phase 3**: Enforce authentication
   - Enable auth middleware to require login
   - Ensure all critical users have accounts
   - Provide grace period with warnings

## Recommended Next Steps

1. **Start Small**: Implement basic form authentication first
2. **Test Thoroughly**: Ensure auth works before enforcing it
3. **Document**: Update your README with login instructions
4. **Monitor**: Add logging for auth events (logins, failed attempts)
5. **Iterate**: Add OAuth, MFA, etc. based on user needs

## Additional Resources

- [Remix Auth Documentation](https://github.com/sergiodxa/remix-auth)
- [React Router Authentication Guide](https://reactrouter.com/how-to/auth)
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [React Router v7 Migration Guide](https://reactrouter.com/upgrading/v6)

## Conclusion

Integrating `remix-auth` into the TensorZero UI is straightforward given the existing architecture. The framework already supports the necessary infrastructure (middleware, context providers, SSR), making it a natural fit.

The recommended approach is:
1. **Form-based authentication** for initial implementation (simplest)
2. **OAuth** for enterprise SSO integration (GitHub, Google, etc.)
3. **Role-based permissions** to complement the existing read-only mode

This will provide a secure, scalable authentication system that integrates cleanly with your existing code patterns.
