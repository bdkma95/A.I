use anchor_lang::prelude::*;
use anchor_spl::token::{self, Mint, TokenAccount, Transfer};
use solana_program::clock::Clock;

declare_id!("YourProgramID");

#[program]
pub mod antivaxxx {
    use super::*;

    // Initialize admin account
    pub fn initialize_admin(ctx: Context<InitializeAdmin>) -> Result<()> {
        let admin = &mut ctx.accounts.admin;
        admin.authority = *ctx.accounts.payer.key;
        admin.bump = *ctx.bumps.get("admin").unwrap();
        Ok(())
    }

    // Initialize vesting schedule with enhanced security
    pub fn initialize_vesting(
        ctx: Context<InitializeVesting>,
        total_supply: u64,
        cliff_duration: i64,
        vesting_duration: i64,
        revocable: bool,
    ) -> Result<()> {
        let vesting = &mut ctx.accounts.vesting_schedule;
        let clock = Clock::get()?;
        
        // Validate input parameters
        require!(cliff_duration >= 0, ErrorCode::InvalidValues);
        require!(vesting_duration > 0, ErrorCode::InvalidValues);
        require!(total_supply > 0, ErrorCode::InvalidValues);
        require!(cliff_duration < vesting_duration, ErrorCode::InvalidValues);

        // Initialize vesting schedule
        vesting.start_time = clock.unix_timestamp;
        vesting.cliff_duration = cliff_duration;
        vesting.vesting_duration = vesting_duration;
        vesting.total_amount = total_supply;
        vesting.released_amount = 0;
        vesting.revocable = revocable;
        vesting.revoked = false;
        vesting.beneficiary = *ctx.accounts.beneficiary.key;
        vesting.mint = ctx.accounts.mint.key();

        // Mint tokens to vesting account
        let seeds = &[b"admin", &[ctx.accounts.admin.bump]];
        let signer = &[&seeds[..]];
        
        token::mint_to(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                token::MintTo {
                    mint: ctx.accounts.mint.to_account_info(),
                    to: ctx.accounts.vesting_vault.to_account_info(),
                    authority: ctx.accounts.admin.to_account_info(),
                },
                signer,
            ),
            total_supply,
        )?;

        Ok(())
    }

    // Release tokens with enhanced security checks
    pub fn release_tokens(ctx: Context<ReleaseTokens>) -> Result<()> {
        let vesting = &mut ctx.accounts.vesting_schedule;
        let clock = Clock::get()?;
        
        // Validate vesting state
        require!(!vesting.revoked, ErrorCode::VestingRevoked);
        require!(vesting.locked == false, ErrorCode::VestingLocked);

        // Calculate releasable amount
        let releasable = vesting.releasable_amount(clock.unix_timestamp)?;
        require!(releasable > 0, ErrorCode::NoTokensAvailable);

        // Validate token balances
        let vault_balance = ctx.accounts.vesting_vault.amount;
        require!(vault_balance >= releasable, ErrorCode::InsufficientBalance);

        // Update vesting state atomically
        vesting.locked = true;
        vesting.released_amount = vesting
            .released_amount
            .checked_add(releasable)
            .ok_or(ErrorCode::Overflow)?;

        // Transfer tokens
        let seeds = &[b"admin", &[ctx.accounts.admin.bump]];
        let signer = &[&seeds[..]];
        
        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.vesting_vault.to_account_info(),
                    to: ctx.accounts.beneficiary_vault.to_account_info(),
                    authority: ctx.accounts.admin.to_account_info(),
                },
                signer,
            ),
            releasable,
        )?;

        // Finalize state update
        vesting.locked = false;

        emit!(ReleaseEvent {
            beneficiary: vesting.beneficiary,
            amount: releasable,
            timestamp: clock.unix_timestamp,
        });

        Ok(())
    }

    // Admin function to revoke vesting
    pub fn revoke_vesting(ctx: Context<RevokeVesting>) -> Result<()> {
        let vesting = &mut ctx.accounts.vesting_schedule;
        
        require!(vesting.revocable, ErrorCode::NotRevocable);
        require!(!vesting.revoked, ErrorCode::AlreadyRevoked);

        // Calculate remaining tokens
        let remaining = vesting.total_amount.checked_sub(vesting.released_amount)
            .ok_or(ErrorCode::Overflow)?;

        // Transfer remaining tokens back to admin
        let seeds = &[b"admin", &[ctx.accounts.admin.bump]];
        let signer = &[&seeds[..]];
        
        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.vesting_vault.to_account_info(),
                    to: ctx.accounts.admin_vault.to_account_info(),
                    authority: ctx.accounts.admin.to_account_info(),
                },
                signer,
            ),
            remaining,
        )?;

        vesting.revoked = true;

        Ok(())
    }
}

#[derive(Accounts)]
pub struct InitializeAdmin<'info> {
    #[account(
        init,
        payer = payer,
        space = 8 + Admin::LEN,
        seeds = [b"admin"],
        bump
    )]
    pub admin: Account<'info, Admin>,
    #[account(mut)]
    pub payer: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct InitializeVesting<'info> {
    #[account(
        init,
        payer = payer,
        space = 8 + VestingSchedule::LEN,
        seeds = [b"vesting", beneficiary.key().as_ref()],
        bump
    )]
    pub vesting_schedule: Account<'info, VestingSchedule>,
    #[account(mut)]
    pub payer: Signer<'info>,
    /// CHECK: Verified by PDA constraint
    pub beneficiary: UncheckedAccount<'info>,
    #[account(
        mut,
        seeds = [b"admin"],
        bump = admin.bump
    )]
    pub admin: Account<'info, Admin>,
    #[account(
        init,
        payer = payer,
        token::mint = mint,
        token::authority = admin,
        seeds = [b"vault", vesting_schedule.key().as_ref()],
        bump
    )]
    pub vesting_vault: Account<'info, TokenAccount>,
    pub mint: Account<'info, Mint>,
    #[account(address = token::ID)]
    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct ReleaseTokens<'info> {
    #[account(
        mut,
        seeds = [b"vesting", beneficiary.key().as_ref()],
        bump
    )]
    pub vesting_schedule: Account<'info, VestingSchedule>,
    #[account(
        mut,
        seeds = [b"vault", vesting_schedule.key().as_ref()],
        bump
    )]
    pub vesting_vault: Account<'info, TokenAccount>,
    #[account(
        mut,
        constraint = beneficiary_vault.owner == beneficiary.key()
    )]
    pub beneficiary_vault: Account<'info, TokenAccount>,
    /// CHECK: Verified by PDA constraint
    pub beneficiary: UncheckedAccount<'info>,
    #[account(
        seeds = [b"admin"],
        bump = admin.bump
    )]
    pub admin: Account<'info, Admin>,
    #[account(address = token::ID)]
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct RevokeVesting<'info> {
    #[account(
        mut,
        seeds = [b"vesting", vesting_schedule.beneficiary.as_ref()],
        bump
    )]
    pub vesting_schedule: Account<'info, VestingSchedule>,
    #[account(
        mut,
        seeds = [b"vault", vesting_schedule.key().as_ref()],
        bump
    )]
    pub vesting_vault: Account<'info, TokenAccount>,
    #[account(
        mut,
        address = admin.authority
    )]
    pub admin_authority: Signer<'info>,
    #[account(
        seeds = [b"admin"],
        bump = admin.bump
    )]
    pub admin: Account<'info, Admin>,
    #[account(mut)]
    pub admin_vault: Account<'info, TokenAccount>,
    #[account(address = token::ID)]
    pub token_program: Program<'info, Token>,
}

#[account]
pub struct Admin {
    pub authority: Pubkey,
    pub bump: u8,
}

#[account]
pub struct VestingSchedule {
    pub start_time: i64,
    pub cliff_duration: i64,
    pub vesting_duration: i64,
    pub total_amount: u64,
    pub released_amount: u64,
    pub beneficiary: Pubkey,
    pub mint: Pubkey,
    pub revocable: bool,
    pub revoked: bool,
    pub locked: bool, // Anti-reentrancy flag
}

impl VestingSchedule {
    pub const LEN: usize = 8 + 8 + 8 + 8 + 8 + 32 + 32 + 1 + 1 + 1;
    
    pub fn releasable_amount(&self, current_time: i64) -> Result<u64> {
        require!(current_time >= self.start_time, ErrorCode::InvalidTime);
        
        let elapsed_time = current_time.checked_sub(self.start_time)
            .ok_or(ErrorCode::Underflow)?;

        // Cliff period check
        if elapsed_time < self.cliff_duration {
            return Ok(0);
        }

        let vesting_time = elapsed_time.checked_sub(self.cliff_duration)
            .ok_or(ErrorCode::Underflow)?;
        
        let total_vesting_time = self.vesting_duration.checked_sub(self.cliff_duration)
            .ok_or(ErrorCode::InvalidValues)?;

        let vested_amount = if vesting_time >= total_vesting_time {
            self.total_amount
        } else {
            self.total_amount
                .checked_mul(vesting_time as u64)
                .and_then(|v| v.checked_div(total_vesting_time as u64))
                .unwrap_or(0)
        };

        Ok(vested_amount.checked_sub(self.released_amount)
            .ok_or(ErrorCode::Underflow)?)
    }
}

#[error_code]
pub enum ErrorCode {
    #[msg("Invalid input parameters")]
    InvalidValues,
    #[msg("Insufficient balance")]
    InsufficientBalance,
    #[msg("No tokens available for release")]
    NoTokensAvailable,
    #[msg("Arithmetic overflow")]
    Overflow,
    #[msg("Arithmetic underflow")]
    Underflow,
    #[msg("Vesting schedule is revoked")]
    VestingRevoked,
    #[msg("Vesting schedule is locked")]
    VestingLocked,
    #[msg("Vesting is not revocable")]
    NotRevocable,
    #[msg("Vesting already revoked")]
    AlreadyRevoked,
    #[msg("Invalid time value")]
    InvalidTime,
}

#[event]
pub struct ReleaseEvent {
    pub beneficiary: Pubkey,
    pub amount: u64,
    pub timestamp: i64,
}
