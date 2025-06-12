
export function safeStringify(value: unknown, space?: string | number): string {
  try {
    // null/undefined
    if (value === null || value === undefined) {
      return String(value);
    }

    // primitives
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
      return String(value);
    }

    // Use a replacer to handle problematic values
    const seen = new WeakSet();
    
    const replacer = (key: string, val: any): any => {
      // Handle circular references
      if (val !== null && typeof val === 'object') {
        if (seen.has(val)) {
          return '[Circular Reference]';
        }
        seen.add(val);
      }

      //  functions
      if (typeof val === 'function') {
        return `[Function: ${val.name || 'anonymous'}]`;
      }

      //  symbols
      if (typeof val === 'symbol') {
        return val.toString();
      }

      //  undefined 
      if (val === undefined) {
        return '[Undefined]';
      }

      //  special objects
      if (val instanceof Date) {
        return val.toISOString();
      }

      if (val instanceof RegExp) {
        return val.toString();
      }

      if (val instanceof Error) {
        return {
          name: val.name,
          message: val.message,
          
        };
      }

      //  BigInt
      if (typeof val === 'bigint') {
        return val.toString() + 'n';
      }

      return val;
    };

    return JSON.stringify(value, replacer,space);
  } catch (error) {
    
    console.warn('JSON.stringify failed', error);
    
    if (typeof value === 'object' && value !== null) {
      try {
        
        const constructor = value.constructor?.name || 'Object';
        const keys = Object.keys(value);
        return `[${constructor} with keys: ${keys.join(', ')}]`;
      } catch {
        return '[Unserializable Object]';
      }
    }
    
    return '[Unserializable Value]';
  }
}
