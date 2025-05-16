const esbuild = require("esbuild");
const fs = require('fs');
const path = require('path');

const production = process.argv.includes('--production');
const watch = process.argv.includes('--watch');

/**
 * @type {import('esbuild').Plugin}
 */
const esbuildProblemMatcherPlugin = {
	name: 'esbuild-problem-matcher',

	setup(build) {
		build.onStart(() => {
			console.log('[watch] build started');
		});
		build.onEnd((result) => {
			result.errors.forEach(({ text, location }) => {
				console.error(`✘ [ERROR] ${text}`);
				console.error(`    ${location.file}:${location.line}:${location.column}:`);
			});
			console.log('[watch] build finished');
		});
	},
};

/**
 * @type {import('esbuild').Plugin}
 */
const copyHtmlPlugin = {
	name: 'copy-html',
	setup(build) {
		build.onEnd(() => {
			// src 디렉토리의 HTML 파일을 dist로 복사
			const srcDir = path.join(__dirname, 'src');
			const distDir = path.join(__dirname, 'dist');

			// dist/src 디렉토리가 없으면 생성
			const distSrcDir = path.join(distDir, 'src');
			if (!fs.existsSync(distSrcDir)) {
				fs.mkdirSync(distSrcDir, { recursive: true });
			}

			// HTML 파일 복사
			const htmlFiles = fs.readdirSync(srcDir).filter(file => file.endsWith('.html'));
			htmlFiles.forEach(file => {
				fs.copyFileSync(
					path.join(srcDir, file),
					path.join(distSrcDir, file)
				);
			});
		});
	},
};

async function main() {
	const ctx = await esbuild.context({
		entryPoints: [
			'src/extension.ts'
		],
		bundle: true,
		format: 'cjs',
		minify: production,
		sourcemap: !production,
		sourcesContent: false,
		platform: 'node',
		outfile: 'dist/extension.js',
		external: ['vscode'],
		logLevel: 'silent',
		plugins: [
			esbuildProblemMatcherPlugin,
			copyHtmlPlugin,
		],
	});
	if (watch) {
		await ctx.watch();
	} else {
		await ctx.rebuild();
		await ctx.dispose();
	}
}

main().catch(e => {
	console.error(e);
	process.exit(1);
});
